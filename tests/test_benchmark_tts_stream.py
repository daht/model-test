import json
import struct
from pathlib import Path

import pytest

from scripts.benchmark_tts_stream import (
    BenchmarkConfig,
    BenchmarkError,
    Observation,
    aggregate_level,
    load_corpus,
    parse_binary_chunk,
    parse_config,
    playback_underrun_seconds,
    render_markdown,
)


def config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        corpus_path=tmp_path / "corpus.jsonl",
        output_dir=tmp_path / "out",
        endpoint="ws://tts/v1/tts/stream",
        api_key="secret",
        model="Fun-CosyVoice3-0.5B-2512",
        voice="default",
        sample_rate=24000,
        transport="binary",
        concurrency_levels=(1,),
        arrival_rates=(),
        duration_seconds=10,
        warmup_requests=1,
        request_timeout_seconds=30,
        max_ttfa_p95_seconds=0.8,
        max_chunk_gap_p99_seconds=0.5,
        max_error_rate=0.001,
        max_underrun_seconds=0,
        random_seed=1,
    )


def success(**overrides) -> Observation:
    values = {
        "mode": "closed",
        "level": 1.0,
        "started_offset_seconds": 0.0,
        "completed_offset_seconds": 2.0,
        "text_id": "zh-short-01",
        "language": "zh",
        "bucket": "short",
        "characters": 10,
        "ttfa_seconds": 0.4,
        "e2e_seconds": 2.0,
        "audio_seconds": 3.0,
        "audio_bytes": 144000,
        "chunks": 4,
        "max_chunk_gap_seconds": 0.3,
        "playback_underrun_seconds": 0.0,
        "error_category": None,
    }
    values.update(overrides)
    return Observation(**values)


def test_parse_binary_chunk_validates_header_and_pcm():
    sequence, offset, pcm = parse_binary_chunk(
        struct.pack("<4sIQ", b"TTS1", 3, 960) + b"\x01\x00\x02\x00"
    )
    assert (sequence, offset, pcm) == (3, 960, b"\x01\x00\x02\x00")

    with pytest.raises(BenchmarkError, match="invalid magic"):
        parse_binary_chunk(struct.pack("<4sIQ", b"NOPE", 0, 0) + b"\0\0")
    with pytest.raises(BenchmarkError, match="odd-length"):
        parse_binary_chunk(struct.pack("<4sIQ", b"TTS1", 0, 0) + b"\0\0\0")


def test_load_corpus_preserves_id_language_and_bucket(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "zh-short-01",
                "language": "zh",
                "bucket": "short",
                "text": "你好",
                "voice": "default",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    records = load_corpus(corpus)
    assert records[0].text_id == "zh-short-01"
    assert records[0].language == "zh"
    assert records[0].bucket == "short"


def test_load_corpus_rejects_duplicate_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    row = json.dumps({"id": "same", "text": "hello"})
    corpus.write_text(row + "\n" + row + "\n")
    with pytest.raises(BenchmarkError, match="unique"):
        load_corpus(corpus)


def test_playback_underrun_uses_accumulated_audio_buffer():
    assert playback_underrun_seconds([1.0, 3.0, 5.5], [3.0, 2.0, 1.0]) == 0
    assert playback_underrun_seconds([1.0, 4.5], [2.0, 1.0]) == 1.5


def test_aggregate_level_reports_streaming_metrics_and_slo(tmp_path):
    result = aggregate_level(
        "closed",
        1,
        10,
        [success(), success(ttfa_seconds=0.6, e2e_seconds=3.0)],
        config(tmp_path),
    )
    assert result.requests_per_second == 0.2
    assert result.audio_seconds_per_second == 0.6
    assert result.inflight_average == 0.5
    assert result.ttfa_p95_seconds == 0.6
    assert result.chunk_gap_p99_seconds == 0.3
    assert result.slo_passed is True


def test_aggregate_level_fails_slo_on_underrun_or_ttfa(tmp_path):
    result = aggregate_level(
        "closed",
        2,
        10,
        [success(ttfa_seconds=1.0, playback_underrun_seconds=0.1)],
        config(tmp_path),
    )
    assert result.playback_underrun_requests == 1
    assert result.slo_passed is False


def test_parse_config_reads_websocket_environment(tmp_path):
    parsed = parse_config(
        [
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--concurrency",
            "1,3",
            "--arrival-rates",
            "0.1,0.25",
        ],
        {
            "API_KEY": "secret",
            "TTS_STREAM_BENCHMARK_URL": "wss://example/v1/tts/stream",
            "TTS_MODEL_NAME": "model",
        },
    )
    assert parsed.endpoint == "wss://example/v1/tts/stream"
    assert parsed.model == "model"
    assert parsed.concurrency_levels == (1, 3)
    assert parsed.arrival_rates == (0.1, 0.25)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--concurrency", "0"),
        ("--concurrency", "1,1"),
        ("--arrival-rates", "nan"),
        ("--duration-seconds", "0"),
        ("--max-error-rate", "2"),
    ],
)
def test_parse_config_rejects_invalid_options(tmp_path, option, value):
    args = [
        "--corpus",
        str(tmp_path / "corpus.jsonl"),
        "--output-dir",
        str(tmp_path / "out"),
        option,
        value,
    ]
    with pytest.raises(BenchmarkError):
        parse_config(args, {"API_KEY": "secret"})


def test_render_markdown_contains_capacity_metrics(tmp_path):
    level = aggregate_level("closed", 1, 10, [success()], config(tmp_path))
    report = type(
        "Report",
        (),
        {
            "model": "model",
            "transport": "binary",
            "sample_rate": 24000,
            "duration_seconds": 10,
            "corpus_records": 1,
            "levels": (level,),
        },
    )()
    markdown = render_markdown(report)
    assert "TTFA" in markdown
    assert "音频 RTFx" in markdown
    assert "gap p95/p99/max" in markdown
    assert "断流请求/秒" in markdown
    assert "| closed | 1 |" in markdown
