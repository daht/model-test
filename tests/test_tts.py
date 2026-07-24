import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.tts import CosyVoiceTTSSynthesizer, _cosyvoice_result_to_wav


def _settings(tmp_path: Path):
    return types.SimpleNamespace(
        tts_cosyvoice_repo=None,
        tts_model_id="/models/CosyVoice",
        tts_prompt_text="prompt",
        tts_prompt_wav=str(tmp_path / "prompt.wav"),
        tts_default_voice="default",
        tts_sample_rate=24000,
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
