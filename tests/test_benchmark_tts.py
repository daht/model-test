import json
import wave
from io import BytesIO
from pathlib import Path

import pytest

from scripts.benchmark_tts import (
    BenchmarkConfig,
    BenchmarkError,
    Observation,
    aggregate_level,
    load_corpus,
    parse_config,
    wav_duration_seconds,
)


def _wav(seconds: int = 1) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * 24000 * seconds)
    return output.getvalue()


def test_wav_duration_is_measured_from_response_audio():
    assert wav_duration_seconds(_wav(2)) == pytest.approx(2.0)


def test_wav_duration_rejects_non_audio():
    with pytest.raises(BenchmarkError):
        wav_duration_seconds(b"not wav")


def test_load_corpus_accepts_text_and_optional_voice(tmp_path: Path):
    path = tmp_path / "tts.jsonl"
    path.write_text(json.dumps({"text": "你好", "voice": "default"}) + "\n", encoding="utf-8")
    records = load_corpus(path)
    assert records[0].text == "你好"
    assert records[0].voice == "default"
    assert records[0].characters == 2


def test_load_corpus_rejects_unknown_fields(tmp_path: Path):
    path = tmp_path / "tts.jsonl"
    path.write_text(json.dumps({"text": "hello", "language": "en"}) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkError):
        load_corpus(path)


def test_aggregate_level_projects_cost_by_audio_seconds():
    config = BenchmarkConfig(Path("corpus"), Path("out"), "http://tts", "key", (1,), 1, 0, 1, 1, 0.01, 100)
    result = aggregate_level(1, 1, [Observation(0.2, 10, 2, 100, None)], config)
    assert result.audio_seconds_per_second == pytest.approx(2)
    assert result.monthly_audio_seconds_capacity == pytest.approx(2 * 30 * 24 * 60 * 60)
    assert result.gpu_cost_per_million_audio_seconds_cny == pytest.approx(100 / (2 * 30 * 24 * 60 * 60) * 1_000_000)


def test_parse_config_uses_tts_endpoint_and_api_key_from_environment(tmp_path: Path):
    config = parse_config(["--corpus", str(tmp_path / "x"), "--output-dir", str(tmp_path / "out")], {"API_KEY": "key", "TTS_BENCHMARK_URL": "http://example/v1/tts"})
    assert config.endpoint == "http://example/v1/tts"
    assert config.api_key == "key"
