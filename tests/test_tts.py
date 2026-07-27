import sys
import threading
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from app.tts import CosyVoiceTTSSynthesizer, _cosyvoice_result_to_wav, create_tts_synthesizer
from app.tts_qwen import Qwen3TTSSynthesizer, _load_qwen_model
from app.tts_triton import TritonTTSSynthesizer, _float_audio_to_pcm


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
                "text": "hello",
                "language": "Chinese",
                "speaker": "Vivian",
                "instruct": None,
            }
            return [np.array([0.25, -0.25], dtype=np.float32)], 24000

    synthesizer = Qwen3TTSSynthesizer(settings)
    synthesizer._model = FakeModel()

    assert list(synthesizer.stream_pcm("hello", "default")) == [
        np.array([8191, -8191], dtype="<i2").tobytes()
    ]


def test_qwen_model_load_falls_back_to_auto_on_meta_tensor_error():
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype):
            calls.append((model_id, device_map, dtype))
            if device_map != "auto":
                raise NotImplementedError("Cannot copy out of meta tensor; no data!")
            return {"model_id": model_id, "device_map": device_map, "dtype": dtype}

    loaded = _load_qwen_model(FakeModel, "/models/qwen", device_map="cuda:0", dtype="bf16")

    assert loaded == {"model_id": "/models/qwen", "device_map": "auto", "dtype": "bf16"}
    assert calls == [
        ("/models/qwen", "cuda:0", "bf16"),
        ("/models/qwen", "auto", "bf16"),
    ]


def test_qwen_model_load_falls_back_to_cpu_then_moves_model_after_meta_errors(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.device = lambda value: value
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    calls = []

    class FakeInnerModel:
        def to(self, device):
            calls.append(("move", device))

    class FakeWrapper:
        def __init__(self):
            self.model = FakeInnerModel()
            self.device = "cpu"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype, **kwargs):
            calls.append((model_id, device_map, dtype, kwargs))
            if device_map in {"cuda:0", "auto"}:
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            return FakeWrapper()

    loaded = _load_qwen_model(FakeModel, "/models/qwen", device_map="cuda:0", dtype="bf16")

    assert loaded.device == "cuda:0"
    assert calls == [
        ("/models/qwen", "cuda:0", "bf16", {}),
        ("/models/qwen", "auto", "bf16", {}),
        ("/models/qwen", None, "bf16", {"low_cpu_mem_usage": False}),
        ("move", "cuda:0"),
    ]


def test_qwen_model_load_falls_back_on_runtime_meta_tensor_error():
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype):
            calls.append((model_id, device_map, dtype))
            if device_map != "auto":
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            return {"model_id": model_id, "device_map": device_map, "dtype": dtype}

    loaded = _load_qwen_model(FakeModel, "/models/qwen", device_map="cuda:0", dtype="bf16")

    assert loaded == {"model_id": "/models/qwen", "device_map": "auto", "dtype": "bf16"}
    assert calls == [
        ("/models/qwen", "cuda:0", "bf16"),
        ("/models/qwen", "auto", "bf16"),
    ]


def test_factory_selects_qwen_without_loading_model(tmp_path):
    settings = _settings(tmp_path)
    settings.tts_backend = "qwen"

    assert isinstance(create_tts_synthesizer(settings), Qwen3TTSSynthesizer)
