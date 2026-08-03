import argparse
import asyncio

import pytest

from scripts import evaluate_asr_samples
from scripts import stream_asr_client


def valid_stream_info():
    return {
        "protocol_version": 2,
        "websocket_url": "/v1/transcribe/stream",
        "format": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
        "api_key": "server-secret",
    }


def valid_backends():
    return {
        "ready": True,
        "workers": {
            "local": {
                "worker_id": "local",
                "model_id": "/models/Qwen3-ASR-0.6B",
                "model_revision": "Qwen3-ASR-0.6B",
                "gpu_id": "cuda:0",
                "lifecycle": "ready",
                "accepting": True,
            }
        },
    }


def evaluation_args(tmp_path, **overrides):
    values = {
        "url": "ws://operator:password@asr.example.test/v1/transcribe/stream?token=url-secret",
        "zh_audio": tmp_path / "zh.wav",
        "en_audio": tmp_path / "en.wav",
        "output": tmp_path / "result.md",
        "chunk_ms": 200,
        "realtime": False,
        "inspect_only": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fake_streaming_server_records_identity_and_both_language_results(
    monkeypatch, tmp_path, capsys
):
    args = evaluation_args(tmp_path)
    args.zh_audio.write_bytes(b"zh sample")
    args.en_audio.write_bytes(b"en sample")
    monkeypatch.setenv("API_KEY", "runtime-secret")
    monkeypatch.setattr(stream_asr_client, "fetch_stream_info", lambda *_: valid_stream_info())
    monkeypatch.setattr(evaluate_asr_samples, "fetch_backend_snapshot", lambda *_: valid_backends())

    async def fake_stream(client_args):
        assert client_args.verify_protocol is True
        assert client_args.api_key_source == "environment"
        text = "你好" if client_args.language == "zh" else "hello"
        return stream_asr_client.StreamResult(
            events=[
                {"type": "ready", "sequence": 1},
                {"type": "sentence_final", "text": text, "sequence": 2},
                {"type": "final", "text": "", "sequence": 3},
            ]
        )

    monkeypatch.setattr(stream_asr_client, "stream_audio", fake_stream)

    asyncio.run(evaluate_asr_samples.run(args))

    report = args.output.read_text()
    assert "Qwen3-ASR-0.6B" in report
    assert '"workers"' in report
    assert "### zh" in report
    assert "### en" in report
    assert "你好" in report
    assert "hello" in report
    assert "runtime-secret" not in report
    assert "server-secret" not in report
    assert "url-secret" not in report
    assert "password" not in report
    assert "ws://asr.example.test/v1/transcribe/stream" in report
    assert "runtime-secret" not in capsys.readouterr().out


def test_gateway_stream_info_without_accepting_model_does_not_write_evaluation_entry(monkeypatch, tmp_path):
    args = evaluation_args(tmp_path)
    monkeypatch.setenv("API_KEY", "runtime-secret")
    monkeypatch.setattr(
        stream_asr_client,
        "fetch_stream_info",
        lambda *_: valid_stream_info(),
    )
    monkeypatch.setattr(evaluate_asr_samples, "fetch_backend_snapshot", lambda *_: {"workers": {}})

    with pytest.raises(evaluate_asr_samples.EvaluationError, match="accepting worker"):
        asyncio.run(evaluate_asr_samples.run(args))

    assert not args.output.exists()


def test_failed_second_stream_does_not_write_successful_evaluation_entry(
    monkeypatch, tmp_path
):
    args = evaluation_args(tmp_path)
    args.zh_audio.write_bytes(b"zh sample")
    args.en_audio.write_bytes(b"en sample")
    monkeypatch.setenv("API_KEY", "runtime-secret")
    monkeypatch.setattr(stream_asr_client, "fetch_stream_info", lambda *_: valid_stream_info())
    monkeypatch.setattr(evaluate_asr_samples, "fetch_backend_snapshot", lambda *_: valid_backends())

    async def fake_stream(client_args):
        if client_args.language == "en":
            raise stream_asr_client.StreamClientError("backend failed")
        return stream_asr_client.StreamResult(
            events=[{"type": "final", "text": "zh", "sequence": 1}]
        )

    monkeypatch.setattr(stream_asr_client, "stream_audio", fake_stream)

    with pytest.raises(evaluate_asr_samples.EvaluationError, match="en stream failed"):
        asyncio.run(evaluate_asr_samples.run(args))

    assert not args.output.exists()
