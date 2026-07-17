import asyncio
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from app.asr_faster_whisper import DecodedText, FasterWhisperAdapter, FasterWhisperEngine
from app.asr_gateway_backends import DispatchMode, ResultMode, StreamingMode, VadMode
from app.asr_gateway_scheduler import BatchKey, InferenceJob
from app.asr_observability import CapacityBufferError


@dataclass
class EngineCall:
    lengths: list[int]
    language: str | None
    beam_size: int


class RecordingEngine:
    def __init__(self):
        self.calls: list[EngineCall] = []
        self.warmups = 0
        self.closed = 0
        self.detected_languages = ["zh", "ja"]

    def warmup(self):
        self.warmups += 1

    def transcribe_batch(self, audio, *, language, beam_size):
        self.calls.append(EngineCall([len(item) for item in audio], language, beam_size))
        if language is None:
            languages = self.detected_languages[: len(audio)]
        else:
            languages = [language] * len(audio)
        return [
            DecodedText(f"{item_language}:{len(waveform)}:{beam_size}", item_language)
            for waveform, item_language in zip(audio, languages)
        ]

    def close(self):
        self.closed += 1


def make_adapter(engine, *, batch_size=4, max_segment_samples=480_000):
    return FasterWhisperAdapter(
        lambda: engine,
        worker_id="local",
        model_id="/models/faster-whisper-large-v3",
        model_revision="large-v3-test-revision",
        gpu_id="cuda:0",
        session_capacity=14,
        batch_size=batch_size,
        partial_beam_size=1,
        final_beam_size=5,
        max_segment_samples=max_segment_samples,
    )


def make_job(session_id, sequence, pcm, *, final=False, language="zh"):
    return InferenceJob(
        job_id=f"{session_id}:{sequence}",
        session_id=session_id,
        generation=1,
        job_sequence=sequence,
        worker_id="local",
        backend_session_id=f"fw-{session_id}",
        start_sample=(sequence - 1) * (len(pcm) // 2),
        end_sample=sequence * (len(pcm) // 2),
        pcm=pcm,
        deadline=1,
        batch_key=BatchKey(
            "local", "large-v3-test-revision", language, "transcribe", False,
            "", "final" if final else "partial", "pcm_s16le", 0,
        ),
        final=final,
    )


def test_engine_suppresses_repeated_three_token_sequences(monkeypatch):
    audio_module = types.ModuleType("faster_whisper.audio")
    audio_module.pad_or_trim = lambda features: features
    tokenizer_module = types.ModuleType("faster_whisper.tokenizer")

    class Tokenizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def decode(self, _tokens):
            return "decoded"

    tokenizer_module.Tokenizer = Tokenizer
    transcribe_module = types.ModuleType("faster_whisper.transcribe")

    class TranscriptionOptions:
        def __init__(self, **values):
            self.__dict__.update(values)

    transcribe_module.TranscriptionOptions = TranscriptionOptions
    transcribe_module.get_suppressed_tokens = lambda _tokenizer, values: values
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio_module)
    monkeypatch.setitem(sys.modules, "faster_whisper.tokenizer", tokenizer_module)
    monkeypatch.setitem(sys.modules, "faster_whisper.transcribe", transcribe_module)

    class Model:
        hf_tokenizer = object()
        model = type("InnerModel", (), {"is_multilingual": True})()

        @staticmethod
        def feature_extractor(_audio):
            return np.zeros((80, 2), dtype=np.float32)

    captured = {}

    class Pipeline:
        @staticmethod
        def generate_segment_batched(_features, _tokenizer, options):
            captured["options"] = options
            return None, [{"tokens": [1]}]

    engine = object.__new__(FasterWhisperEngine)
    engine._model = Model()
    engine._pipeline = Pipeline()

    result = engine.transcribe_batch(
        [np.zeros(1600, dtype=np.float32)],
        language="zh",
        beam_size=1,
    )

    assert result == [DecodedText("decoded", "zh")]
    assert captured["options"].no_repeat_ngram_size == 3


def test_adapter_advertises_real_rolling_dynamic_batch_contract():
    adapter = make_adapter(RecordingEngine())

    caps = adapter.capabilities

    assert caps.streaming_mode is StreamingMode.ROLLING
    assert caps.dispatch_mode is DispatchMode.DYNAMIC_MICROBATCH
    assert caps.result_mode is ResultMode.REPLACEABLE_SEGMENT
    assert caps.vad_mode is VadMode.GATEWAY
    assert caps.max_batch_items == 4
    assert caps.tasks == ("transcribe",)
    assert caps.languages == ("auto",)


def test_cross_session_partial_is_one_batch_and_pcm_stays_isolated():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        await adapter.open_session("b", language="zh")
        first = await adapter.submit([
            make_job("a", 1, b"\xff\x7f" * 3),
            make_job("b", 1, b"\x00\x80" * 5),
        ])
        second = await adapter.submit([
            make_job("a", 2, b"\x01\x00" * 2),
            make_job("b", 2, b"\x02\x00" * 4),
        ])
        snapshot = await adapter.snapshot()
        await adapter.close()
        return engine, first, second, snapshot

    engine, first, second, snapshot = asyncio.run(scenario())

    assert engine.warmups == 1 and engine.closed == 1
    assert engine.calls == [
        EngineCall([3, 5], "zh", 1),
        EngineCall([5, 9], "zh", 1),
    ]
    assert [item.session_id for item in first] == ["a", "b"]
    assert [item.text for item in second] == ["zh:5:1", "zh:9:1"]
    assert snapshot["active_sessions"] == 2


def test_final_batch_uses_beam_five_then_control_consumes_cache_without_decode():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        await adapter.open_session("b", language="zh")
        results = await adapter.submit([
            make_job("a", 1, b"\x01\x00" * 3, final=True),
            make_job("b", 1, b"\x02\x00" * 4, final=True),
        ])
        a_final = await adapter.finish_segment("a")
        b_final = await adapter.finish_session("b")
        snapshot = await adapter.snapshot()
        return engine.calls, results, a_final, b_final, snapshot

    calls, results, a_final, b_final, snapshot = asyncio.run(scenario())

    assert calls == [EngineCall([3, 4], "zh", 5)]
    assert [result.text for result in results] == ["zh:3:5", "zh:4:5"]
    assert a_final.text == "zh:3:5" and b_final.text == "zh:4:5"
    assert snapshot["session_audio_samples"] == 0
    assert snapshot["active_sessions"] == 1


def test_mixed_partial_final_submission_partitions_beams_and_restores_order():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        observed = []
        adapter.set_engine_observer(lambda **values: observed.append(values))
        await adapter.warmup()
        await adapter.open_session("partial", language="zh")
        await adapter.open_session("final", language="zh")
        results = await adapter.submit([
            make_job("partial", 1, b"\x01\x00" * 3),
            make_job("final", 1, b"\x02\x00" * 4, final=True),
        ])
        return engine.calls, observed, results

    calls, observed, results = asyncio.run(scenario())

    assert calls == [
        EngineCall([3], "zh", 1),
        EngineCall([4], "zh", 5),
    ]
    assert [item["group_count"] for item in observed] == [2, 2]
    assert [item.session_id for item in results] == ["partial", "final"]


def test_explicit_segment_redecodes_one_session_with_final_beam_and_clears_audio():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        await adapter.open_session("a", language="en")
        await adapter.submit([make_job("a", 1, b"\x01\x00" * 6, language="en")])
        final = await adapter.finish_segment("a")
        snapshot = await adapter.snapshot()
        return engine.calls, final, snapshot

    calls, final, snapshot = asyncio.run(scenario())

    assert calls == [EngineCall([6], "en", 1), EngineCall([6], "en", 5)]
    assert final.text == "en:6:5"
    assert snapshot["session_audio_samples"] == 0


def test_auto_language_detects_once_then_partitions_locked_languages_and_restores_order():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        await adapter.open_session("a", language="auto")
        await adapter.open_session("b", language="auto")
        first = await adapter.submit([
            make_job("a", 1, b"\x01\x00" * 3, language="auto"),
            make_job("b", 1, b"\x02\x00" * 4, language="auto"),
        ])
        second = await adapter.submit([
            make_job("b", 2, b"\x03\x00" * 2, language="auto"),
            make_job("a", 2, b"\x04\x00" * 2, language="auto"),
        ])
        return engine.calls, first, second

    calls, first, second = asyncio.run(scenario())

    assert calls == [
        EngineCall([3, 4], None, 1),
        EngineCall([6], "ja", 1),
        EngineCall([5], "zh", 1),
    ]
    assert [item.text for item in first] == ["zh:3:1", "ja:4:1"]
    assert [item.session_id for item in second] == ["b", "a"]
    assert [item.text for item in second] == ["ja:6:1", "zh:5:1"]


def test_stale_identity_rejection_abort_and_close_remove_all_session_state():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        stale = make_job("a", 1, b"\x01\x00")
        stale = InferenceJob(**{**stale.__dict__, "backend_session_id": "wrong"})
        with pytest.raises(KeyError, match="stale session backend identity"):
            await adapter.submit([stale])
        await adapter.abort_session("a")
        await adapter.close()
        return await adapter.snapshot(), engine.closed

    snapshot, closed = asyncio.run(scenario())

    assert snapshot["active_sessions"] == 0
    assert snapshot["session_audio_samples"] == 0
    assert closed == 1


def test_engine_observer_records_actual_group_and_accumulated_audio():
    async def scenario():
        adapter = make_adapter(RecordingEngine())
        observed = []
        adapter.set_engine_observer(lambda **values: observed.append(values))
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        await adapter.open_session("b", language="zh")
        await adapter.submit([
            make_job("a", 1, b"\x01\x00" * 3),
            make_job("b", 1, b"\x02\x00" * 5),
        ])
        return observed

    observed = asyncio.run(scenario())

    assert len(observed) == 1
    assert observed[0]["group_size"] == 2
    assert observed[0]["final"] is False
    assert observed[0]["accumulated_audio_seconds"] == 8 / 16_000
    assert observed[0]["output_characters"] > 0


def test_adapter_utterance_overflow_has_exact_capacity_reason():
    async def scenario():
        adapter = make_adapter(RecordingEngine(), max_segment_samples=24_000)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        await adapter.submit([make_job("a", 1, b"\x01\x00" * 23_999)])
        with pytest.raises(CapacityBufferError, match="utterance") as rejected:
            await adapter.submit([make_job("a", 2, b"\x01\x00" * 2)])
        return rejected.value

    rejected = asyncio.run(scenario())

    assert rejected.reason == "adapter_utterance_limit"
    assert rejected.safe_fields == {"limit": 24_000, "current": 23_999, "incoming": 2}
