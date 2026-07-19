import asyncio
import json
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from app.asr_sensevoice import (
    SenseVoiceAdapter,
    SenseVoiceBatchFailure,
    SenseVoiceDecoded,
    SenseVoiceEngine,
    normalize_sensevoice_output,
)
from app.asr_gateway_backends import DispatchMode, ResultMode, StreamingMode, VadMode
from app.asr_gateway_scheduler import BatchKey, InferenceJob
from app.asr_observability import CapacityBufferError


def install_fake_funasr(monkeypatch, outputs):
    module = types.ModuleType("funasr")

    class Model:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.generate_calls = []

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return outputs

    model = Model()
    module.AutoModel = lambda **kwargs: (setattr(model, "init_kwargs", kwargs) or model)
    monkeypatch.setitem(sys.modules, "funasr", module)
    return model


def test_normalize_sensevoice_output_extracts_tags_and_clean_text():
    decoded = normalize_sensevoice_output(
        "<|zh|><|NEUTRAL|><|Speech|>今天天气不错。"
    )
    assert decoded == SenseVoiceDecoded(
        "今天天气不错。",
        {"language": "zh", "emotion": "neutral", "audio_event": "speech"},
    )


def test_normalize_sensevoice_output_omits_unknown_tags():
    decoded = normalize_sensevoice_output("<|xx|><|EMO_UNKNOWN|>hello")
    assert decoded.text == "hello"
    assert decoded.metadata == {}


def test_engine_uses_local_batch_api_and_preserves_order(monkeypatch, tmp_path):
    model = install_fake_funasr(monkeypatch, [
        {"text": "<|zh|><|NEUTRAL|><|Speech|>甲"},
        {"text": "<|zh|><|HAPPY|><|Laughter|>乙"},
    ])
    engine = SenseVoiceEngine(str(tmp_path), device="cuda:0", use_itn=True)
    result = engine.transcribe_batch(
        [np.zeros(1600, dtype=np.float32), np.ones(800, dtype=np.float32)],
        language="zh",
    )
    assert [item.text for item in result] == ["甲", "乙"]
    assert model.init_kwargs == {
        "model": str(tmp_path), "device": "cuda:0",
        "trust_remote_code": False, "disable_update": True,
    }
    assert model.generate_calls[0]["language"] == "zh"
    assert model.generate_calls[0]["use_itn"] is True
    assert model.generate_calls[0]["batch_size"] == 2


def test_engine_classifies_generate_failure_without_private_text(monkeypatch, tmp_path):
    class PrivateEngineError(RuntimeError):
        pass

    model = install_fake_funasr(monkeypatch, [])

    def fail(**_kwargs):
        raise PrivateEngineError("private-model-details")

    model.generate = fail
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=True)
    with pytest.raises(SenseVoiceBatchFailure) as failure:
        engine.transcribe_batch([np.ones(16, dtype=np.float32)], language="zh")
    assert failure.value.stage == "engine_generate"
    assert failure.value.exception_type == "PrivateEngineError"
    assert "private-model-details" not in str(failure.value)


def test_engine_classifies_result_count_failure(monkeypatch, tmp_path):
    install_fake_funasr(monkeypatch, [{"text": "one"}])
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=True)
    with pytest.raises(SenseVoiceBatchFailure) as failure:
        engine.transcribe_batch(
            [np.ones(16, dtype=np.float32), np.ones(16, dtype=np.float32)],
            language="zh",
        )
    assert failure.value.stage == "result_count"


def test_engine_classifies_result_contract_failure(monkeypatch, tmp_path):
    install_fake_funasr(monkeypatch, [{"private": "raw-output"}])
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=True)
    with pytest.raises(SenseVoiceBatchFailure) as failure:
        engine.transcribe_batch([np.ones(16, dtype=np.float32)], language="zh")
    assert failure.value.stage == "result_contract"
    assert "raw-output" not in str(failure.value)


def test_adapter_emits_one_safe_engine_group_failure(monkeypatch):
    class PrivateEngineError(RuntimeError):
        pass

    class Engine(RecordingEngine):
        def __init__(self):
            super().__init__()
            self.warmed = False

        def transcribe_batch(self, audio, *, language):
            if not self.warmed:
                self.warmed = True
                return super().transcribe_batch(audio, language=language)
            raise SenseVoiceBatchFailure("engine_generate", "PrivateEngineError") from (
                PrivateEngineError("private-model-details")
            )

    class Emitter:
        slow_engine_seconds = 2.0

        def __init__(self):
            self.records = []

        def emit(self, event, *, component, **fields):
            self.records.append({"event": event, "component": component, **fields})

    async def scenario():
        emitter = Emitter()
        monkeypatch.setattr("app.asr_sensevoice.events", lambda: emitter)
        adapter = make_adapter(Engine(), batch_size=2)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        await adapter.open_session("b", language="zh")
        with pytest.raises(SenseVoiceBatchFailure):
            await adapter.submit([
                make_job("a", 1, b"\x01\x00" * 16),
                make_job("b", 1, b"\x02\x00" * 16),
            ])
        return [
            record for record in emitter.records
            if record["event"] == "asr_engine_group_failed"
        ]

    failures = asyncio.run(scenario())
    assert len(failures) == 1
    failure = failures[0]
    assert failure["failure_stage"] == "engine_generate"
    assert failure["exception_type"] == "PrivateEngineError"
    assert failure["group_size"] == 2
    assert failure["final_items"] == 0
    assert failure["accumulated_audio_seconds"] == pytest.approx(0.002)
    assert failure["min_input_audio_seconds"] == pytest.approx(0.001)
    assert failure["max_input_audio_seconds"] == pytest.approx(0.001)
    assert "private-model-details" not in json.dumps(failure)


def test_adapter_classifies_omitted_result(monkeypatch):
    class Engine(RecordingEngine):
        def __init__(self):
            super().__init__()
            self.warmed = False

        def transcribe_batch(self, audio, *, language):
            if not self.warmed:
                self.warmed = True
                return super().transcribe_batch(audio, language=language)
            return [None]

    class Emitter:
        slow_engine_seconds = 2.0

        def __init__(self):
            self.records = []

        def emit(self, event, *, component, **fields):
            self.records.append({"event": event, "component": component, **fields})

    async def scenario():
        emitter = Emitter()
        monkeypatch.setattr("app.asr_sensevoice.events", lambda: emitter)
        adapter = make_adapter(Engine(), batch_size=1)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        with pytest.raises(SenseVoiceBatchFailure) as failure:
            await adapter.submit([make_job("a", 1, b"\x01\x00" * 16)])
        return failure.value, emitter.records

    failure, records = asyncio.run(scenario())
    assert failure.stage == "result_omitted"
    failed = [item for item in records if item["event"] == "asr_engine_group_failed"]
    assert len(failed) == 1
    assert failed[0]["failure_stage"] == "result_omitted"


@pytest.mark.parametrize("outputs", [[], [{"text": "ok"}], [{"other": "x"}]])
def test_engine_result_contract_fails_closed_without_raw_output(monkeypatch, tmp_path, outputs):
    install_fake_funasr(monkeypatch, outputs)
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=False)
    audio = [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)]
    with pytest.raises(SenseVoiceBatchFailure, match="SenseVoice batch failed at") as failure:
        engine.transcribe_batch(audio, language=None)
    assert repr(outputs) not in str(failure.value)


def test_engine_warmup_decodes_bundled_real_speech_to_pcm(monkeypatch, tmp_path):
    sample = tmp_path / "example" / "en.mp3"
    sample.parent.mkdir()
    sample.write_bytes(b"not-decoded-by-fake")
    model = install_fake_funasr(monkeypatch, [
        {"text": "<|en|><|NEUTRAL|><|Speech|>hello"},
    ])
    librosa = types.ModuleType("librosa")
    librosa.load = lambda *_args, **_kwargs: (
        np.array([0.0, 0.25, -0.25], dtype=np.float32),
        16_000,
    )
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=True)
    pcm = engine.warmup()
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 8192, -8192]
    assert model.generate_calls == []


def test_adapter_warmup_runs_real_pcm_through_live_submit_path(monkeypatch, tmp_path):
    sample = tmp_path / "example" / "en.mp3"
    sample.parent.mkdir()
    sample.write_bytes(b"decoded-by-fake-librosa")
    model = install_fake_funasr(monkeypatch, [
        {"text": "<|en|><|NEUTRAL|><|Speech|>hello"},
    ])
    librosa = types.ModuleType("librosa")
    librosa.load = lambda *_args, **_kwargs: (
        np.array([0.0, 0.25, -0.25], dtype=np.float32),
        16_000,
    )
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=True)

    async def scenario():
        adapter = SenseVoiceAdapter(
            lambda: engine,
            worker_id="local", model_id=str(tmp_path), model_revision="rev",
            gpu_id="cpu", session_capacity=1, batch_size=1,
            max_segment_samples=16_000,
        )
        await adapter.warmup()
        return await adapter.snapshot()

    snapshot = asyncio.run(scenario())
    submitted = model.generate_calls[0]["input"][0]
    assert isinstance(submitted, np.ndarray)
    assert submitted.dtype == np.float32
    assert submitted.tolist() == [0.0, 0.25, -0.25]
    assert model.generate_calls[0]["language"] == "en"
    assert snapshot["active_sessions"] == 0
    assert snapshot["session_audio_samples"] == 0


def test_engine_warmup_rejects_missing_sample(monkeypatch, tmp_path):
    install_fake_funasr(monkeypatch, [])
    engine = SenseVoiceEngine(str(tmp_path), device="cpu", use_itn=True)
    with pytest.raises(RuntimeError, match="warmup speech sample"):
        engine.warmup()


@dataclass
class EngineCall:
    lengths: list[int]
    language: str | None


class RecordingEngine:
    def __init__(self):
        self.calls = []
        self.warmups = 0
        self.closed = 0

    def warmup(self):
        self.warmups += 1
        return b"\x01\x00" * 2

    def transcribe_batch(self, audio, *, language):
        self.calls.append(EngineCall([len(item) for item in audio], language))
        effective = language or "en"
        return [
            SenseVoiceDecoded(
                f"text-{len(item)}",
                {"language": effective, "emotion": "neutral"},
            )
            for item in audio
        ]

    def close(self):
        self.closed += 1


def make_adapter(engine, *, batch_size=8, max_segment_samples=240_000, manifest=None):
    return SenseVoiceAdapter(
        lambda: engine,
        worker_id="local",
        model_id="/models/SenseVoiceSmall",
        model_revision="sensevoice-test-revision",
        gpu_id="cuda:0",
        session_capacity=64,
        batch_size=batch_size,
        max_segment_samples=max_segment_samples,
        model_manifest_path=manifest,
    )


def make_job(session_id, sequence, pcm, *, start=0, final=False, language="zh"):
    samples = len(pcm) // 2
    return InferenceJob(
        job_id=f"{session_id}:{sequence}",
        session_id=session_id,
        generation=1,
        job_sequence=sequence,
        worker_id="local",
        backend_session_id=f"sv-{session_id}",
        start_sample=start,
        end_sample=start + samples,
        pcm=pcm,
        deadline=1,
        batch_key=BatchKey(
            "local", "sensevoice-test-revision", language, "transcribe", False,
            "", "final" if final else "partial", "pcm_s16le", 0,
        ),
        final=final,
    )


def test_adapter_contract_and_open_validation():
    async def scenario():
        adapter = make_adapter(RecordingEngine())
        await adapter.warmup()
        assert adapter.capabilities.streaming_mode is StreamingMode.ROLLING
        assert adapter.capabilities.dispatch_mode is DispatchMode.DYNAMIC_MICROBATCH
        assert adapter.capabilities.result_mode is ResultMode.REPLACEABLE_SEGMENT
        assert adapter.capabilities.vad_mode is VadMode.GATEWAY
        assert adapter.capabilities.languages == ("auto", "zh", "yue", "en", "ja", "ko")
        for options in ({"task": "translate"}, {"timestamps": True}, {"language": "fr"}):
            with pytest.raises(ValueError):
                await adapter.open_session("bad", **options)

    asyncio.run(scenario())


def test_partial_redecodes_full_accumulated_utterance_and_replaces_text():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        engine.calls.clear()
        await adapter.open_session("a", language="zh")
        first = await adapter.submit([make_job("a", 1, b"\x01\x00" * 3)])
        second = await adapter.submit([make_job("a", 2, b"\x02\x00" * 2, start=3)])
        return engine.calls, first, second

    calls, first, second = asyncio.run(scenario())
    assert calls == [EngineCall([3], "zh"), EngineCall([5], "zh")]
    assert [first[0].text, second[0].text] == ["text-3", "text-5"]


def test_cross_session_batch_preserves_identity_and_metadata():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        engine.calls.clear()
        await adapter.open_session("a", language="zh")
        await adapter.open_session("b", language="ja")
        return await adapter.submit([
            make_job("a", 1, b"\x01\x00" * 3, language="zh"),
            make_job("b", 1, b"\x02\x00" * 4, language="ja"),
        ])

    results = asyncio.run(scenario())
    assert [item.session_id for item in results] == ["a", "b"]
    assert [item.metadata["language"] for item in results] == ["zh", "ja"]


def test_final_batch_is_cached_then_finish_clears_pcm_and_metadata():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        engine.calls.clear()
        await adapter.open_session("a", language="zh")
        final_result = (await adapter.submit([
            make_job("a", 1, b"\x01\x00" * 3, final=True),
        ]))[0]
        finish = await adapter.finish_segment("a")
        snapshot = await adapter.snapshot()
        return engine.calls, final_result, finish, snapshot

    calls, final_result, finish, snapshot = asyncio.run(scenario())
    assert calls == [EngineCall([3], "zh")]
    assert finish.text == final_result.text
    assert finish.metadata == final_result.metadata
    assert snapshot["session_audio_samples"] == 0


def test_explicit_segment_fallback_decodes_full_audio_and_finish_removes_session():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        await adapter.warmup()
        engine.calls.clear()
        await adapter.open_session("a", language="en")
        await adapter.submit([make_job("a", 1, b"\x01\x00" * 4, language="en")])
        segment = await adapter.finish_segment("a")
        await adapter.submit([make_job("a", 2, b"\x02\x00" * 2, start=4, language="en")])
        final = await adapter.finish_session("a")
        return engine.calls, segment, final, await adapter.snapshot()

    calls, segment, final, snapshot = asyncio.run(scenario())
    assert calls == [
        EngineCall([4], "en"), EngineCall([4], "en"),
        EngineCall([2], "en"), EngineCall([2], "en"),
    ]
    assert segment.text == "text-4" and final.text == "text-2"
    assert snapshot["active_sessions"] == 0


def test_validation_failures_do_not_extend_pcm_and_count_mismatch_fails_closed():
    class WrongCountEngine(RecordingEngine):
        def transcribe_batch(self, audio, *, language):
            warmup = super().transcribe_batch(audio, language=language)
            if language == "en":
                return warmup
            return []

    async def scenario():
        adapter = make_adapter(WrongCountEngine(), max_segment_samples=3)
        await adapter.warmup()
        await adapter.open_session("a", language="zh")
        stale = make_job("a", 1, b"\x01\x00")
        stale = InferenceJob(**{**stale.__dict__, "backend_session_id": "wrong"})
        with pytest.raises(KeyError, match="stale session backend identity"):
            await adapter.submit([stale])
        with pytest.raises(ValueError, match="two jobs"):
            await adapter.submit([make_job("a", 1, b"\x01\x00"), make_job("a", 2, b"\x02\x00")])
        with pytest.raises(CapacityBufferError, match="utterance"):
            await adapter.submit([make_job("a", 1, b"\x01\x00" * 4)])
        assert (await adapter.snapshot())["session_audio_samples"] == 0
        with pytest.raises(SenseVoiceBatchFailure, match="result_count"):
            await adapter.submit([make_job("a", 1, b"\x01\x00" * 2)])
        retained = await adapter.snapshot()
        await adapter.cancel("a:1")
        await adapter.abort_session("a")
        return retained, await adapter.snapshot()

    retained, cleaned = asyncio.run(scenario())
    assert retained["session_audio_samples"] == 2
    assert cleaned["session_audio_samples"] == 0


def test_abort_close_and_engine_observer_leave_no_session_audio():
    async def scenario():
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        observed = []
        adapter.set_engine_observer(lambda **values: observed.append(values))
        await adapter.warmup()
        observed.clear()
        await adapter.open_session("a", language="zh")
        await adapter.submit([make_job("a", 1, b"\x01\x00" * 3)])
        await adapter.abort_session("a")
        await adapter.open_session("b", language="zh")
        await adapter.close()
        return observed, await adapter.snapshot(), engine.closed

    observed, snapshot, closed = asyncio.run(scenario())
    assert observed[0]["group_size"] == 1
    assert observed[0]["accumulated_audio_seconds"] == 3 / 16000
    assert snapshot["session_audio_samples"] == 0
    assert closed == 1


def test_warmup_verifies_manifest_before_constructing_engine(monkeypatch):
    order = []
    monkeypatch.setattr(
        "app.asr_sensevoice.verify_model_manifest",
        lambda *_args: order.append("manifest"),
    )
    engine = RecordingEngine()

    async def scenario():
        adapter = SenseVoiceAdapter(
            lambda: (order.append("engine") or engine),
            worker_id="local", model_id="/model", model_revision="rev",
            gpu_id="cuda:0", session_capacity=1, batch_size=1,
            max_segment_samples=10, model_manifest_path="/manifest.json",
        )
        await adapter.warmup()

    asyncio.run(scenario())
    assert order == ["manifest", "engine"]
