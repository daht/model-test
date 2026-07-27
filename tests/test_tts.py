import sys
import threading
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from app.tts import CosyVoiceTTSSynthesizer, _cosyvoice_result_to_wav, create_tts_synthesizer
from app.tts_qwen import Qwen3TTSSynthesizer, _load_qwen_model
from app.tts_triton import TritonTTSSynthesizer, _float_audio_to_pcm
from app.tts_vllm_omni import VLLMOmniTTSSynthesizer


def _settings(tmp_path: Path):
    return types.SimpleNamespace(
        tts_cosyvoice_repo=None,
        tts_model_id="/models/CosyVoice",
        tts_prompt_text="prompt",
        tts_prompt_wav=str(tmp_path / "prompt.wav"),
        tts_default_voice="default",
        tts_sample_rate=24000,
        tts_triton_url="127.0.0.1:18001",
        tts_triton_model_name="cosyvoice3",
        tts_qwen_batch_size=4,
        tts_qwen_batch_wait_ms=50,
        tts_qwen_queue_size=8,
        tts_qwen_request_timeout_seconds=2.0,
        tts_vllm_omni_base_url="http://vllm-omni:8091",
        tts_vllm_omni_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        tts_vllm_omni_timeout_seconds=30.0,
        tts_vllm_omni_api_key=None,
    )


def test_cosyvoice3_is_selected_by_official_automodel_and_zero_shot_registered(monkeypatch, tmp_path):
    calls = {}

    class FakeModel:
        def add_zero_shot_spk(self, text, wav, speaker):
            calls["speaker"] = (text, wav, speaker)
            return True

        def inference_zero_shot(self, text, prompt_text, prompt_wav, **kwargs):
            calls["inference"] = (text, prompt_text, prompt_wav, kwargs)
            return [{"tts_speech": np.array([[0.0, 0.5, -0.5]], dtype=np.float32)}]

    fake_module = types.ModuleType("cosyvoice.cli.cosyvoice")
    fake_module.AutoModel = lambda **kwargs: (calls.setdefault("model_dir", kwargs), FakeModel())[1]
    monkeypatch.setitem(sys.modules, "cosyvoice", types.ModuleType("cosyvoice"))
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", types.ModuleType("cosyvoice.cli"))
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", fake_module)

    synthesizer = CosyVoiceTTSSynthesizer(_settings(tmp_path))
    audio = synthesizer.synthesize("hello")

    assert calls["model_dir"] == {"model_dir": "/models/CosyVoice"}
    assert calls["speaker"][2] == "default"
    assert calls["inference"][3]["zero_shot_spk_id"] == "default"
    assert calls["inference"][3]["stream"] is True
    assert audio.startswith(b"RIFF")


def test_cosyvoice_stream_pcm_yields_before_inference_finishes(tmp_path):
    events = []

    class FakeModel:
        def inference_zero_shot(self, *args, **kwargs):
            assert kwargs["stream"] is True
            events.append("started")
            yield {"tts_speech": np.array([[0.25]], dtype=np.float32)}
            events.append("continued")
            yield {"tts_speech": np.array([[-0.25]], dtype=np.float32)}
            events.append("finished")

    synthesizer = CosyVoiceTTSSynthesizer(_settings(tmp_path))
    synthesizer._model = FakeModel()
    stream = synthesizer.stream_pcm("hello")

    first = next(stream)

    assert first == np.array([0.25 * 32767], dtype="<i2").tobytes()
    assert events == ["started"]
    assert list(stream)
    assert events == ["started", "continued", "finished"]


def test_unknown_voice_is_rejected():
    synthesizer = object.__new__(CosyVoiceTTSSynthesizer)
    synthesizer.settings = types.SimpleNamespace(tts_default_voice="default")
    synthesizer._model = object()
    with pytest.raises(RuntimeError, match="unknown TTS voice"):
        synthesizer._select_inference("hello", "other")


def test_cosyvoice_float_chunks_are_converted_to_pcm_and_concatenated():
    audio = _cosyvoice_result_to_wav(
        [
            {"tts_speech": np.array([[0.5]], dtype=np.float32)},
            {"tts_speech": np.array([[-0.5]], dtype=np.float32)},
        ],
        sample_rate=24000,
    )
    assert audio.startswith(b"RIFF")
    assert len(audio) == 48


def test_triton_float_audio_is_converted_to_pcm_s16le():
    assert _float_audio_to_pcm(np.array([1.0, -1.0, 0.5], dtype=np.float32)) == np.array(
        [32767, -32767, 16383], dtype="<i2"
    ).tobytes()


def test_triton_inputs_match_cosyvoice_model_contract(tmp_path):
    class FakeInput:
        def __init__(self, name, shape, dtype):
            self.value = (name, shape, dtype, None)

        def set_data_from_numpy(self, value):
            self.value = (*self.value[:3], value)

    class FakeGrpc:
        InferInput = FakeInput

    settings = _settings(tmp_path)
    synthesizer = TritonTTSSynthesizer(settings)
    inputs = synthesizer._inputs(FakeGrpc, np.array([0.1, -0.1], dtype=np.float32), "hello")

    assert [item.value[:3] for item in inputs] == [
        ("reference_wav", (1, 2), "FP32"),
        ("reference_wav_len", (1, 1), "INT32"),
        ("reference_text", [1, 1], "BYTES"),
        ("target_text", [1, 1], "BYTES"),
    ]
    assert inputs[1].value[3].tolist() == [[2]]
    assert inputs[3].value[3].tolist() == [["hello"]]


def test_triton_stream_consumes_decoupled_chunks_until_final_response(tmp_path):
    with wave.open(str(tmp_path / "prompt.wav"), "wb") as prompt:
        prompt.setnchannels(1)
        prompt.setsampwidth(2)
        prompt.setframerate(16000)
        prompt.writeframes(b"\0\0" * 160)

    class FakeInput:
        def __init__(self, name, shape, dtype):
            self.name = name

        def set_data_from_numpy(self, value):
            pass

    class FakeResponse:
        def __init__(self, audio=None, final=False):
            self._audio = audio
            self.parameters = {
                "triton_final_response": type("Flag", (), {"bool_param": final})()
            }

        def get_response(self):
            return self

        def as_numpy(self, name):
            return self._audio

    class FakeClient:
        def __init__(self, url):
            self.callback = None

        def start_stream(self, callback):
            self.callback = callback

        def async_stream_infer(self, **kwargs):
            threading.Thread(
                target=lambda: (
                    self.callback(FakeResponse(np.array([0.25], dtype=np.float32)), None),
                    self.callback(FakeResponse(np.array([-0.25], dtype=np.float32)), None),
                    self.callback(FakeResponse(final=True), None),
                ),
                daemon=True,
            ).start()

        def stop_stream(self, cancel_requests=True):
            pass

    class FakeGrpc:
        InferInput = FakeInput
        InferRequestedOutput = lambda name: name
        InferenceServerClient = FakeClient

    settings = _settings(tmp_path)
    synthesizer = TritonTTSSynthesizer(settings)
    synthesizer._grpcclient = lambda: FakeGrpc

    assert list(synthesizer.stream_pcm("hello")) == [
        np.array([0.25 * 32767], dtype="<i2").tobytes(),
        np.array([-0.25 * 32767], dtype="<i2").tobytes(),
    ]


def test_qwen_custom_voice_uses_deployment_settings(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_qwen_language = "Chinese"
    settings.tts_qwen_speaker = "Vivian"
    settings.tts_qwen_instruct = ""

    class FakeModel:
        def generate_custom_voice(self, **kwargs):
            assert kwargs == {
                "text": ["hello"],
                "language": "Chinese",
                "speaker": "Vivian",
                "instruct": None,
            }
            return [np.array([0.25, -0.25], dtype=np.float32)], 24000

    synthesizer = Qwen3TTSSynthesizer(settings)
    synthesizer._model = FakeModel()
    try:
        assert list(synthesizer.stream_pcm("hello", "default")) == [
            np.array([8191, -8191], dtype="<i2").tobytes()
        ]
    finally:
        synthesizer.close()


def test_qwen_model_load_uses_official_device_map():
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype):
            calls.append((model_id, device_map, dtype))
            return object()

    loaded = _load_qwen_model(FakeModel, "/models/qwen", device_map="cuda:0", dtype="bf16")

    assert loaded is not None
    assert calls == [("/models/qwen", "cuda:0", "bf16")]


def test_qwen_concurrent_requests_share_one_explicit_batch(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_qwen_language = "Chinese"
    settings.tts_qwen_speaker = "Vivian"
    settings.tts_qwen_instruct = ""
    settings.tts_qwen_batch_wait_ms = 200
    calls = []

    class FakeModel:
        def generate_custom_voice(self, **kwargs):
            calls.append(kwargs)
            return [
                np.array([0.1 if text == "one" else 0.2], dtype=np.float32)
                for text in kwargs["text"]
            ], 24000

    synthesizer = Qwen3TTSSynthesizer(settings)
    synthesizer._model = FakeModel()
    barrier = threading.Barrier(3)
    results = {}

    def synthesize(text):
        barrier.wait()
        results[text] = list(synthesizer.stream_pcm(text, "default"))

    threads = [threading.Thread(target=synthesize, args=(text,)) for text in ("one", "two")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    try:
        assert all(not thread.is_alive() for thread in threads)
        assert len(calls) == 1
        assert sorted(calls[0]["text"]) == ["one", "two"]
        assert set(results) == {"one", "two"}
    finally:
        synthesizer.close()


def test_qwen_model_load_is_single_owner_under_concurrency(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    settings.tts_qwen_language = "Chinese"
    settings.tts_qwen_speaker = "Vivian"
    settings.tts_qwen_instruct = ""
    settings.tts_device = "cuda:0"
    settings.torch_dtype = "bfloat16"
    calls = []

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"

    class FakeQwenModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            time.sleep(0.05)
            return object()

    fake_qwen = types.ModuleType("qwen_tts")
    fake_qwen.Qwen3TTSModel = FakeQwenModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen)

    synthesizer = Qwen3TTSSynthesizer(settings)
    threads = [threading.Thread(target=synthesizer._load_model) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert len(calls) == 1


def test_vllm_omni_streams_pcm_chunks_and_uses_custom_voice_contract(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_qwen_language = "Chinese"
    settings.tts_qwen_speaker = "Vivian"
    settings.tts_qwen_instruct = ""
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"\x01\x00"
            yield b"\x02\x00"

    class FakeClient:
        def stream(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return FakeResponse()

    synthesizer = VLLMOmniTTSSynthesizer(settings, client=FakeClient())

    assert list(synthesizer.stream_pcm("hello", "default")) == [b"\x01\x00", b"\x02\x00"]
    assert calls == [
        (
            "POST",
            "http://vllm-omni:8091/v1/audio/speech",
            {
                "json": {
                    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "input": "hello",
                    "voice": "Vivian",
                    "response_format": "pcm",
                    "stream": True,
                    "task_type": "CustomVoice",
                    "language": "Chinese",
                },
                "headers": {},
                "timeout": 30.0,
            },
        )
    ]


def test_vllm_omni_rejects_invalid_pcm_chunks(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_qwen_language = "Chinese"
    settings.tts_qwen_speaker = "Vivian"
    settings.tts_qwen_instruct = ""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"\x01"

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeResponse()

    synthesizer = VLLMOmniTTSSynthesizer(settings, client=FakeClient())

    with pytest.raises(RuntimeError, match="odd-length pcm_s16le"):
        list(synthesizer.stream_pcm("hello"))


def test_vllm_omni_reassembles_pcm_samples_split_by_http_chunks(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_qwen_language = "Chinese"
    settings.tts_qwen_speaker = "Vivian"
    settings.tts_qwen_instruct = ""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"\x01"
            yield b"\x00\x02"
            yield b"\x00"

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeResponse()

    synthesizer = VLLMOmniTTSSynthesizer(settings, client=FakeClient())

    assert list(synthesizer.stream_pcm("hello")) == [b"\x01\x00", b"\x02\x00"]


def test_vllm_omni_start_checks_upstream_health(tmp_path):
    settings = _settings(tmp_path)
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    synthesizer = VLLMOmniTTSSynthesizer(settings, client=FakeClient())
    synthesizer.start()

    assert calls == [
        (
            "http://vllm-omni:8091/health",
            {"timeout": 30.0},
        )
    ]


def test_factory_selects_qwen_without_loading_model(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_backend = "qwen"

    assert isinstance(create_tts_synthesizer(settings), Qwen3TTSSynthesizer)


def test_factory_selects_vllm_omni(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_backend = "vllm_omni"

    assert isinstance(create_tts_synthesizer(settings), VLLMOmniTTSSynthesizer)
