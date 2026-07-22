"""HTTP TTS capacity and GPU-cost benchmark.

The benchmark measures real WAV audio duration returned by the service.  This
is intentionally separate from the MT benchmark because TTS cost is naturally
expressed per generated audio second (and not per request or input character).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import httpx

MONTH_SECONDS = 30 * 24 * 60 * 60


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusRecord:
    text: str
    voice: str | None
    characters: int


@dataclass(frozen=True)
class Observation:
    latency_seconds: float
    characters: int
    audio_seconds: float
    audio_bytes: int
    error_category: str | None


@dataclass(frozen=True)
class LevelResult:
    concurrency: int
    measured_seconds: float
    attempted_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    requests_per_second: float
    characters_per_second: float
    audio_seconds: float
    audio_seconds_per_second: float
    audio_hours_per_second: float
    latency_min_seconds: float
    latency_mean_seconds: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    latency_max_seconds: float
    error_categories: dict[str, int]
    monthly_audio_seconds_capacity: float | None
    gpu_cost_per_million_audio_seconds_cny: float | None
    slo_passed: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    corpus_path: Path
    output_dir: Path
    endpoint: str
    api_key: str
    concurrency_levels: tuple[int, ...]
    duration_seconds: float
    warmup_requests: int
    request_timeout_seconds: float
    max_p95_seconds: float
    max_error_rate: float
    a10_monthly_cost_cny: float


@dataclass(frozen=True)
class BenchmarkReport:
    duration_seconds: float
    max_p95_seconds: float
    max_error_rate: float
    a10_monthly_cost_cny: float
    record_count: int
    levels: tuple[LevelResult, ...]
    selected_sustainable: LevelResult | None


def load_corpus(path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict) or set(value) - {"text", "voice"}:
                raise BenchmarkError(f"line {line_number}: expected text and optional voice")
            text = value.get("text")
            voice = value.get("voice")
            if not isinstance(text, str) or not text.strip():
                raise BenchmarkError(f"line {line_number}: text must not be blank")
            if voice is not None and (not isinstance(voice, str) or not voice.strip()):
                raise BenchmarkError(f"line {line_number}: voice must be a non-blank string")
            records.append(CorpusRecord(text=text.strip(), voice=voice, characters=len(text.strip())))
    if not records:
        raise BenchmarkError("corpus has no records")
    return records


def wav_duration_seconds(payload: bytes) -> float:
    try:
        with wave.open(__import__("io").BytesIO(payload), "rb") as audio:
            frame_rate = audio.getframerate()
            frames = audio.getnframes()
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        raise BenchmarkError("response is not a valid WAV") from exc
    if frame_rate <= 0 or frames <= 0:
        raise BenchmarkError("WAV has no positive duration")
    return frames / frame_rate


async def send_request(client: httpx.AsyncClient, config: BenchmarkConfig, record: CorpusRecord) -> Observation:
    started = time.perf_counter()
    try:
        body = {"text": record.text}
        if record.voice is not None:
            body["voice"] = record.voice
        response = await client.post(config.endpoint, headers={"X-API-Key": config.api_key}, json=body)
        if response.status_code != 200:
            raise BenchmarkError(f"http_{response.status_code}")
        audio_seconds = wav_duration_seconds(response.content)
        return Observation(time.perf_counter() - started, record.characters, audio_seconds, len(response.content), None)
    except httpx.TimeoutException:
        category = "timeout"
    except httpx.RequestError:
        category = "connection_error"
    except BenchmarkError as exc:
        category = str(exc).split()[0]
    return Observation(time.perf_counter() - started, 0, 0.0, 0, category)


async def run_level(client: httpx.AsyncClient, config: BenchmarkConfig, records: Sequence[CorpusRecord], concurrency: int) -> tuple[list[Observation], float]:
    if concurrency <= 0 or config.duration_seconds <= 0:
        raise BenchmarkError("concurrency and duration must be positive")
    next_index = 0
    gate = asyncio.Event()
    started = time.perf_counter()
    deadline = started + config.duration_seconds

    async def worker() -> list[Observation]:
        nonlocal next_index
        await gate.wait()
        result = []
        while time.perf_counter() < deadline:
            record = records[next_index % len(records)]
            next_index += 1
            result.append(await send_request(client, config, record))
        return result

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    gate.set()
    results = await asyncio.gather(*tasks)
    return [item for group in results for item in group], time.perf_counter() - started


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(percentile / 100 * len(values)) - 1)]


def aggregate_level(concurrency: int, measured: float, observations: Sequence[Observation], config: BenchmarkConfig) -> LevelResult:
    successful = [item for item in observations if item.error_category is None]
    attempted = len(observations)
    error_rate = (attempted - len(successful)) / attempted if attempted else 1.0
    latencies = [item.latency_seconds for item in observations]
    audio_seconds = sum(item.audio_seconds for item in successful)
    monthly = audio_seconds / measured * MONTH_SECONDS if audio_seconds else None
    cost = config.a10_monthly_cost_cny * 1_000_000 / monthly if monthly else None
    return LevelResult(
        concurrency, measured, attempted, len(successful), attempted - len(successful), error_rate,
        len(successful) / measured, sum(i.characters for i in successful) / measured,
        audio_seconds, audio_seconds / measured, audio_seconds / measured / 3600,
        min(latencies, default=0), statistics.fmean(latencies) if latencies else 0,
        _percentile(latencies, 50), _percentile(latencies, 95), _percentile(latencies, 99), max(latencies, default=0),
        dict(sorted(Counter(i.error_category for i in observations if i.error_category).items())),
        monthly, cost, bool(successful and _percentile(latencies, 95) <= config.max_p95_seconds and error_rate <= config.max_error_rate),
    )


def parse_config(argv: Sequence[str] | None = None, environ: dict[str, str] | None = None) -> BenchmarkConfig:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Benchmark the HTTP TTS service")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--duration-seconds", default=30.0, type=float)
    parser.add_argument("--warmup-requests", default=3, type=int)
    parser.add_argument("--request-timeout-seconds", default=30.0, type=float)
    parser.add_argument("--max-p95-seconds", default=1.0, type=float)
    parser.add_argument("--max-error-rate", default=0.001, type=float)
    parser.add_argument("--a10-monthly-cost-cny", default=2132.72, type=float)
    args = parser.parse_args(argv)
    try:
        levels = tuple(int(item) for item in args.concurrency.split(","))
        if not levels or any(item <= 0 for item in levels) or len(set(levels)) != len(levels):
            raise ValueError
        numeric = (args.duration_seconds, args.request_timeout_seconds, args.max_p95_seconds, args.a10_monthly_cost_cny)
        if not all(math.isfinite(item) and item > 0 for item in numeric):
            raise ValueError
        if args.warmup_requests < 0 or not 0 <= args.max_error_rate <= 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("invalid benchmark numeric option") from exc
    api_key = args.api_key or env.get("API_KEY")
    endpoint = env.get("TTS_BENCHMARK_URL", "http://127.0.0.1:8003/v1/tts")
    if not api_key or not endpoint.strip():
        raise BenchmarkError("API key and benchmark endpoint are required")
    return BenchmarkConfig(args.corpus, args.output_dir, endpoint, api_key, levels, args.duration_seconds, args.warmup_requests, args.request_timeout_seconds, args.max_p95_seconds, args.max_error_rate, args.a10_monthly_cost_cny)


def select_sustainable_level(levels: Sequence[LevelResult]) -> LevelResult | None:
    passing = [level for level in levels if level.slo_passed]
    return max(passing, key=lambda item: item.concurrency) if passing else None


def render_markdown(report: BenchmarkReport) -> str:
    lines = ["# TTS 容量与成本压测报告", "", "成本仅包含独立单张 A10 GPU，不包含 CPU、网络、存储和运维。", "", f"- 语料记录数：{report.record_count}", f"- A10 月成本：{report.a10_monthly_cost_cny:.2f} 元", f"- SLO：P95 ≤ {report.max_p95_seconds:.3f} 秒，错误率 ≤ {report.max_error_rate:.4%}", "", "| 并发 | 成功/请求 | 请求/秒 | 音频秒/秒 | 音频小时/秒 | P50/P95/P99 秒 | 错误率 | 每百万音频秒 GPU 成本 | SLO |", "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | :---:|"]
    for level in report.levels:
        cost = "-" if level.gpu_cost_per_million_audio_seconds_cny is None else f"{level.gpu_cost_per_million_audio_seconds_cny:.4f}"
        lines.append(f"| {level.concurrency} | {level.successful_requests}/{level.attempted_requests} | {level.requests_per_second:.3f} | {level.audio_seconds_per_second:.3f} | {level.audio_hours_per_second:.6f} | {level.latency_p50_seconds:.3f}/{level.latency_p95_seconds:.3f}/{level.latency_p99_seconds:.3f} | {level.error_rate:.4%} | {cost} | {'通过' if level.slo_passed else '失败'} |")
    selected = report.selected_sustainable
    lines.extend(["", "可持续容量：" + (f"并发 {selected.concurrency}，每百万音频秒 {selected.gpu_cost_per_million_audio_seconds_cny:.4f} 元。" if selected else "没有并发档满足 SLO。"), "", "公式：月音频秒 = 音频秒/秒 × 2,592,000；每百万音频秒 GPU 成本 = A10 月成本 × 1,000,000 ÷ 月音频秒。", ""])
    return "\n".join(lines)


async def run_benchmark(config: BenchmarkConfig, records: Sequence[CorpusRecord]) -> BenchmarkReport:
    async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
        for index in range(config.warmup_requests):
            observation = await send_request(client, config, records[index % len(records)])
            if observation.error_category:
                raise BenchmarkError(f"warmup failed: {observation.error_category}")
        levels = []
        for concurrency in config.concurrency_levels:
            observations, measured = await run_level(client, config, records, concurrency)
            levels.append(aggregate_level(concurrency, measured, observations, config))
    levels_tuple = tuple(levels)
    return BenchmarkReport(config.duration_seconds, config.max_p95_seconds, config.max_error_rate, config.a10_monthly_cost_cny, len(records), levels_tuple, select_sustainable_level(levels_tuple))


def write_reports(config: BenchmarkConfig, report: BenchmarkReport) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    (config.output_dir / "tts-benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (config.output_dir / "tts-benchmark.md").write_text(render_markdown(report), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_config(argv)
        report = asyncio.run(run_benchmark(config, load_corpus(config.corpus_path)))
        write_reports(config, report)
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"TTS benchmark failed: {exc}", file=sys.stderr)
        return 2
    print("TTS benchmark reports written: tts-benchmark.json, tts-benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
