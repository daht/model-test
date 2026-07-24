"""Benchmark the MiniMax-style WebSocket TTS streaming endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import struct
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import websockets


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusRecord:
    text_id: str
    text: str
    voice: str | None
    language: str
    bucket: str
    characters: int


@dataclass(frozen=True)
class BenchmarkConfig:
    corpus_path: Path
    output_dir: Path
    endpoint: str
    api_key: str
    model: str
    voice: str
    sample_rate: int
    transport: str
    concurrency_levels: tuple[int, ...]
    arrival_rates: tuple[float, ...]
    duration_seconds: float
    warmup_requests: int
    request_timeout_seconds: float
    max_ttfa_p95_seconds: float
    max_chunk_gap_p99_seconds: float
    max_error_rate: float
    max_underrun_seconds: float
    random_seed: int


@dataclass(frozen=True)
class Observation:
    mode: str
    level: float
    started_offset_seconds: float
    completed_offset_seconds: float
    text_id: str
    language: str
    bucket: str
    characters: int
    ttfa_seconds: float | None
    e2e_seconds: float
    audio_seconds: float
    audio_bytes: int
    chunks: int
    max_chunk_gap_seconds: float | None
    playback_underrun_seconds: float | None
    error_category: str | None


@dataclass(frozen=True)
class LevelResult:
    mode: str
    target: float
    measured_seconds: float
    attempted_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    requests_per_second: float
    audio_seconds: float
    audio_seconds_per_second: float
    inflight_average: float
    ttfa_p50_seconds: float | None
    ttfa_p95_seconds: float | None
    ttfa_p99_seconds: float | None
    e2e_p50_seconds: float | None
    e2e_p95_seconds: float | None
    e2e_p99_seconds: float | None
    chunk_gap_p95_seconds: float | None
    chunk_gap_p99_seconds: float | None
    chunk_gap_max_seconds: float | None
    playback_underrun_requests: int
    playback_underrun_seconds: float
    error_categories: dict[str, int]
    slo_passed: bool


@dataclass(frozen=True)
class BenchmarkReport:
    endpoint: str
    model: str
    voice: str
    sample_rate: int
    transport: str
    duration_seconds: float
    corpus_records: int
    levels: tuple[LevelResult, ...]


def parse_binary_chunk(payload: bytes) -> tuple[int, int, bytes]:
    if len(payload) < 18:
        raise BenchmarkError("binary frame is shorter than the TTS1 header and one sample")
    magic, sequence, sample_offset = struct.unpack("<4sIQ", payload[:16])
    pcm = payload[16:]
    if magic != b"TTS1":
        raise BenchmarkError("binary frame has invalid magic")
    if len(pcm) % 2:
        raise BenchmarkError("binary frame has odd-length pcm_s16le")
    return sequence, sample_offset, pcm


def load_corpus(path: Path) -> list[CorpusRecord]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise BenchmarkError(f"line {line_number}: record must be an object")
            unsupported = set(payload) - {"id", "text", "voice", "language", "bucket"}
            if unsupported:
                raise BenchmarkError(f"line {line_number}: unsupported fields")
            text = payload.get("text")
            voice = payload.get("voice")
            text_id = payload.get("id", f"line-{line_number}")
            language = payload.get("language", "unknown")
            bucket = payload.get("bucket", "unspecified")
            values = {"id": text_id, "language": language, "bucket": bucket}
            if not isinstance(text, str) or not text.strip():
                raise BenchmarkError(f"line {line_number}: text must be nonblank")
            if voice is not None and (not isinstance(voice, str) or not voice.strip()):
                raise BenchmarkError(f"line {line_number}: voice must be nonblank")
            if any(not isinstance(value, str) or not value.strip() for value in values.values()):
                raise BenchmarkError(f"line {line_number}: id, language, and bucket must be nonblank")
            records.append(
                CorpusRecord(
                    text_id=text_id.strip(),
                    text=text.strip(),
                    voice=voice.strip() if voice else None,
                    language=language.strip(),
                    bucket=bucket.strip(),
                    characters=len(text.strip()),
                )
            )
    if not records:
        raise BenchmarkError("corpus has no records")
    if len({record.text_id for record in records}) != len(records):
        raise BenchmarkError("corpus ids must be unique")
    return records


def playback_underrun_seconds(arrivals: Sequence[float], durations: Sequence[float]) -> float:
    if not arrivals or len(arrivals) != len(durations):
        raise BenchmarkError("chunk arrivals and durations must be nonempty and aligned")
    buffer_seconds = 0.0
    underrun_seconds = 0.0
    previous_arrival = arrivals[0]
    for index, (arrival, duration) in enumerate(zip(arrivals, durations)):
        if duration <= 0:
            raise BenchmarkError("audio chunk duration must be positive")
        if index:
            gap = arrival - previous_arrival
            if gap < 0:
                raise BenchmarkError("chunk arrivals must be monotonic")
            underrun_seconds += max(0.0, gap - buffer_seconds)
            buffer_seconds = max(0.0, buffer_seconds - gap)
        buffer_seconds += duration
        previous_arrival = arrival
    return underrun_seconds


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]


async def send_request(
    config: BenchmarkConfig,
    record: CorpusRecord,
    mode: str,
    level: float,
    benchmark_started: float,
) -> Observation:
    started = time.perf_counter()
    arrivals: list[float] = []
    durations: list[float] = []
    audio_bytes = 0
    expected_sequence = 0
    expected_offset = 0
    error_category = None

    try:
        async with asyncio.timeout(config.request_timeout_seconds):
            async with websockets.connect(
                config.endpoint,
                additional_headers={"Authorization": f"Bearer {config.api_key}"},
                max_size=None,
                open_timeout=min(15.0, config.request_timeout_seconds),
                close_timeout=5,
            ) as websocket:
                connected = _json_message(await websocket.recv())
                _require_event(connected, "connected_success")
                await websocket.send(
                    json.dumps(
                        {
                            "event": "task_start",
                            "model": config.model,
                            "voice_setting": {"voice_id": record.voice or config.voice},
                            "audio_setting": {
                                "sample_rate": config.sample_rate,
                                "format": "pcm",
                                "channel": 1,
                            },
                            "stream_options": {"audio_transport": config.transport},
                        }
                    )
                )
                _require_event(_json_message(await websocket.recv()), "task_started")

                synthesis_started = time.perf_counter()
                await websocket.send(
                    json.dumps(
                        {"event": "task_continue", "text": record.text},
                        ensure_ascii=False,
                    )
                )
                await websocket.send(json.dumps({"event": "task_finish"}))

                while True:
                    message = await websocket.recv()
                    received = time.perf_counter()
                    if isinstance(message, bytes):
                        sequence, sample_offset, pcm = parse_binary_chunk(message)
                    else:
                        payload = _json_message(message)
                        if payload.get("event") == "task_failed":
                            status = payload.get("base_resp", {}).get("status_code", "unknown")
                            raise BenchmarkError(f"server_{status}")
                        if payload.get("event") == "task_finished":
                            extra = payload.get("extra_info")
                            if not isinstance(extra, dict):
                                raise BenchmarkError("task_finished is missing extra_info")
                            if extra.get("chunks") != expected_sequence:
                                raise BenchmarkError("server chunk count mismatch")
                            if extra.get("total_samples") != expected_offset:
                                raise BenchmarkError("server sample count mismatch")
                            break
                        _require_event(payload, "task_continued")
                        extra = payload.get("extra_info")
                        data = payload.get("data")
                        if not isinstance(extra, dict) or not isinstance(data, dict):
                            raise BenchmarkError("task_continued metadata is missing")
                        try:
                            sequence = int(extra["chunk_sequence"])
                            sample_offset = int(extra["sample_offset"])
                            pcm = bytes.fromhex(data["audio"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise BenchmarkError("invalid hex audio chunk") from exc

                    if sequence != expected_sequence:
                        raise BenchmarkError("non-contiguous chunk sequence")
                    if sample_offset != expected_offset:
                        raise BenchmarkError("non-contiguous sample offset")
                    if not pcm or len(pcm) % 2:
                        raise BenchmarkError("invalid pcm audio chunk")
                    arrivals.append(received - synthesis_started)
                    duration = len(pcm) / 2 / config.sample_rate
                    durations.append(duration)
                    audio_bytes += len(pcm)
                    expected_sequence += 1
                    expected_offset += len(pcm) // 2
    except TimeoutError:
        error_category = "timeout"
    except websockets.exceptions.InvalidStatus as exc:
        error_category = f"websocket_http_{exc.response.status_code}"
    except websockets.exceptions.WebSocketException:
        error_category = "websocket_error"
    except (BenchmarkError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        error_category = str(exc).split()[0]

    completed = time.perf_counter()
    e2e = completed - started
    if error_category is not None or not arrivals:
        return Observation(
            mode,
            level,
            started - benchmark_started,
            completed - benchmark_started,
            record.text_id,
            record.language,
            record.bucket,
            record.characters,
            None,
            e2e,
            0.0,
            0,
            0,
            None,
            None,
            error_category or "no_audio",
        )

    gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
    return Observation(
        mode,
        level,
        started - benchmark_started,
        completed - benchmark_started,
        record.text_id,
        record.language,
        record.bucket,
        record.characters,
        arrivals[0],
        e2e,
        sum(durations),
        audio_bytes,
        len(arrivals),
        max(gaps, default=0.0),
        playback_underrun_seconds(arrivals, durations),
        None,
    )


def _json_message(message: str | bytes) -> dict:
    if not isinstance(message, str):
        raise BenchmarkError("expected JSON text frame")
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise BenchmarkError("expected JSON object")
    return payload


def _require_event(payload: dict, expected: str) -> None:
    event = payload.get("event")
    if event == "task_failed":
        status = payload.get("base_resp", {}).get("status_code", "unknown")
        raise BenchmarkError(f"server_{status}")
    if event != expected:
        raise BenchmarkError(f"expected_{expected}")


async def run_closed_level(
    config: BenchmarkConfig,
    records: Sequence[CorpusRecord],
    concurrency: int,
) -> tuple[list[Observation], float]:
    benchmark_started = time.perf_counter()
    deadline = benchmark_started + config.duration_seconds
    next_index = 0

    async def worker() -> list[Observation]:
        nonlocal next_index
        observations = []
        while time.perf_counter() < deadline:
            index = next_index
            next_index += 1
            observations.append(
                await send_request(
                    config,
                    records[index % len(records)],
                    "closed",
                    float(concurrency),
                    benchmark_started,
                )
            )
        return observations

    groups = await asyncio.gather(*(worker() for _ in range(concurrency)))
    return [item for group in groups for item in group], time.perf_counter() - benchmark_started


async def run_open_level(
    config: BenchmarkConfig,
    records: Sequence[CorpusRecord],
    arrival_rate: float,
) -> tuple[list[Observation], float]:
    benchmark_started = time.perf_counter()
    deadline = benchmark_started + config.duration_seconds
    rng = random.Random(config.random_seed + round(arrival_rate * 1000))
    tasks = []
    scheduled = benchmark_started
    index = 0
    while scheduled < deadline:
        delay = 0.0 if index == 0 else rng.expovariate(arrival_rate)
        scheduled += delay
        if scheduled >= deadline:
            break
        await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
        record = records[index % len(records)]
        tasks.append(
            asyncio.create_task(
                send_request(config, record, "open", arrival_rate, benchmark_started)
            )
        )
        index += 1
    observations = await asyncio.gather(*tasks) if tasks else []
    return list(observations), time.perf_counter() - benchmark_started


def aggregate_level(
    mode: str,
    target: float,
    measured: float,
    observations: Sequence[Observation],
    config: BenchmarkConfig,
) -> LevelResult:
    successful = [item for item in observations if item.error_category is None]
    attempted = len(observations)
    failed = attempted - len(successful)
    error_rate = failed / attempted if attempted else 1.0
    ttfa = [item.ttfa_seconds for item in successful if item.ttfa_seconds is not None]
    e2e = [item.e2e_seconds for item in successful]
    gaps = [
        item.max_chunk_gap_seconds
        for item in successful
        if item.max_chunk_gap_seconds is not None
    ]
    underruns = [
        item.playback_underrun_seconds
        for item in successful
        if item.playback_underrun_seconds is not None
    ]
    audio_seconds = sum(item.audio_seconds for item in successful)
    total_request_seconds = sum(item.e2e_seconds for item in observations)
    ttfa_p95 = _percentile(ttfa, 95)
    gap_p99 = _percentile(gaps, 99)
    underrun_total = sum(underruns)
    slo_passed = bool(
        successful
        and error_rate <= config.max_error_rate
        and ttfa_p95 is not None
        and ttfa_p95 <= config.max_ttfa_p95_seconds
        and gap_p99 is not None
        and gap_p99 <= config.max_chunk_gap_p99_seconds
        and underrun_total <= config.max_underrun_seconds
    )
    return LevelResult(
        mode=mode,
        target=target,
        measured_seconds=measured,
        attempted_requests=attempted,
        successful_requests=len(successful),
        failed_requests=failed,
        error_rate=error_rate,
        requests_per_second=len(successful) / measured if measured else 0.0,
        audio_seconds=audio_seconds,
        audio_seconds_per_second=audio_seconds / measured if measured else 0.0,
        inflight_average=total_request_seconds / measured if measured else 0.0,
        ttfa_p50_seconds=_percentile(ttfa, 50),
        ttfa_p95_seconds=ttfa_p95,
        ttfa_p99_seconds=_percentile(ttfa, 99),
        e2e_p50_seconds=_percentile(e2e, 50),
        e2e_p95_seconds=_percentile(e2e, 95),
        e2e_p99_seconds=_percentile(e2e, 99),
        chunk_gap_p95_seconds=_percentile(gaps, 95),
        chunk_gap_p99_seconds=gap_p99,
        chunk_gap_max_seconds=max(gaps, default=None),
        playback_underrun_requests=sum(value > 0 for value in underruns),
        playback_underrun_seconds=underrun_total,
        error_categories=dict(
            sorted(Counter(item.error_category for item in observations if item.error_category).items())
        ),
        slo_passed=slo_passed,
    )


def _parse_int_levels(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise BenchmarkError("concurrency levels must be integers") from exc
    if any(item <= 0 for item in levels) or len(set(levels)) != len(levels):
        raise BenchmarkError("concurrency levels must be unique positive integers")
    return levels


def _parse_float_levels(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    try:
        levels = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise BenchmarkError("arrival rates must be numbers") from exc
    if any(not math.isfinite(item) or item <= 0 for item in levels):
        raise BenchmarkError("arrival rates must be positive finite numbers")
    if len(set(levels)) != len(levels):
        raise BenchmarkError("arrival rates must be unique")
    return levels


def parse_config(
    argv: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> BenchmarkConfig:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Benchmark WebSocket streaming TTS")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--transport", choices=("binary", "hex"), default="binary")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--arrival-rates", default="")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-ttfa-p95-seconds", type=float, default=0.8)
    parser.add_argument("--max-chunk-gap-p99-seconds", type=float, default=0.5)
    parser.add_argument("--max-error-rate", type=float, default=0.001)
    parser.add_argument("--max-underrun-seconds", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=20260724)
    args = parser.parse_args(argv)

    concurrency = _parse_int_levels(args.concurrency)
    arrival_rates = _parse_float_levels(args.arrival_rates)
    if not concurrency and not arrival_rates:
        raise BenchmarkError("at least one concurrency or arrival-rate level is required")
    positive = (
        args.duration_seconds,
        args.request_timeout_seconds,
        args.max_ttfa_p95_seconds,
        args.max_chunk_gap_p99_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise BenchmarkError("benchmark durations and SLO thresholds must be positive")
    if args.sample_rate <= 0 or args.warmup_requests < 0 or args.random_seed < 0:
        raise BenchmarkError("invalid integer option")
    if not 0 <= args.max_error_rate <= 1 or args.max_underrun_seconds < 0:
        raise BenchmarkError("invalid SLO option")

    endpoint = args.url or env.get(
        "TTS_STREAM_BENCHMARK_URL",
        "ws://127.0.0.1:8003/v1/tts/stream",
    )
    api_key = args.api_key or env.get("API_KEY")
    model = args.model or env.get("TTS_MODEL_NAME", "Fun-CosyVoice3-0.5B-2512")
    if not api_key or not endpoint.strip() or not model.strip() or not args.voice.strip():
        raise BenchmarkError("URL, API key, model, and voice are required")
    return BenchmarkConfig(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        voice=args.voice,
        sample_rate=args.sample_rate,
        transport=args.transport,
        concurrency_levels=concurrency,
        arrival_rates=arrival_rates,
        duration_seconds=args.duration_seconds,
        warmup_requests=args.warmup_requests,
        request_timeout_seconds=args.request_timeout_seconds,
        max_ttfa_p95_seconds=args.max_ttfa_p95_seconds,
        max_chunk_gap_p99_seconds=args.max_chunk_gap_p99_seconds,
        max_error_rate=args.max_error_rate,
        max_underrun_seconds=args.max_underrun_seconds,
        random_seed=args.random_seed,
    )


async def run_benchmark(
    config: BenchmarkConfig,
    records: Sequence[CorpusRecord],
) -> tuple[BenchmarkReport, list[Observation]]:
    warmup_started = time.perf_counter()
    for index in range(config.warmup_requests):
        observation = await send_request(
            config,
            records[index % len(records)],
            "warmup",
            1.0,
            warmup_started,
        )
        if observation.error_category:
            raise BenchmarkError(f"warmup failed: {observation.error_category}")

    all_observations = []
    levels = []
    for concurrency in config.concurrency_levels:
        observations, measured = await run_closed_level(config, records, concurrency)
        all_observations.extend(observations)
        levels.append(
            aggregate_level("closed", float(concurrency), measured, observations, config)
        )
    for arrival_rate in config.arrival_rates:
        observations, measured = await run_open_level(config, records, arrival_rate)
        all_observations.extend(observations)
        levels.append(
            aggregate_level("open", arrival_rate, measured, observations, config)
        )
    return (
        BenchmarkReport(
            endpoint=config.endpoint,
            model=config.model,
            voice=config.voice,
            sample_rate=config.sample_rate,
            transport=config.transport,
            duration_seconds=config.duration_seconds,
            corpus_records=len(records),
            levels=tuple(levels),
        ),
        all_observations,
    )


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# WebSocket 流式 TTS 压测报告",
        "",
        f"- 模型：`{report.model}`",
        f"- 传输：`{report.transport}` / {report.sample_rate} Hz mono pcm_s16le",
        f"- 单档目标时长：{report.duration_seconds:.1f} 秒",
        f"- 语料数：{report.corpus_records}",
        "",
        "| 模式 | 目标 | 成功/请求 | RPS | 音频 RTFx | 在途均值 | TTFA p50/p95/p99 | E2E p50/p95/p99 | gap p95/p99/max | 断流请求/秒 | 错误率 | SLO |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|:---:|",
    ]
    for level in report.levels:
        lines.append(
            f"| {level.mode} | {level.target:g} | "
            f"{level.successful_requests}/{level.attempted_requests} | "
            f"{level.requests_per_second:.3f} | {level.audio_seconds_per_second:.3f} | "
            f"{level.inflight_average:.3f} | "
            f"{_format(level.ttfa_p50_seconds)}/{_format(level.ttfa_p95_seconds)}/{_format(level.ttfa_p99_seconds)} | "
            f"{_format(level.e2e_p50_seconds)}/{_format(level.e2e_p95_seconds)}/{_format(level.e2e_p99_seconds)} | "
            f"{_format(level.chunk_gap_p95_seconds)}/{_format(level.chunk_gap_p99_seconds)}/{_format(level.chunk_gap_max_seconds)} | "
            f"{level.playback_underrun_requests}/{level.playback_underrun_seconds:.3f} | "
            f"{level.error_rate:.3%} | {'通过' if level.slo_passed else '失败'} |"
        )
    lines.extend(
        [
            "",
            "`audio RTFx = 成功音频总秒数 / 测量墙钟秒数`；在途均值由请求总占用时间除以测量墙钟时间估算。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    config: BenchmarkConfig,
    report: BenchmarkReport,
    observations: Sequence[Observation],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    safe_report = asdict(report)
    safe_report["endpoint"] = report.endpoint.split("?", 1)[0]
    (config.output_dir / "tts-stream-benchmark.json").write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "tts-stream-observations.jsonl").write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in observations),
        encoding="utf-8",
    )
    (config.output_dir / "tts-stream-benchmark.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_config(argv)
        records = load_corpus(config.corpus_path)
        report, observations = asyncio.run(run_benchmark(config, records))
        write_reports(config, report, observations)
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"TTS stream benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(f"TTS stream benchmark reports written to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
