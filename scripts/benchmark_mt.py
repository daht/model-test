import asyncio
import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import httpx


MONTH_SECONDS = 30 * 24 * 60 * 60
_CORPUS_FIELDS = {"source_lang", "target_lang", "text"}


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusRecord:
    source_lang: str
    target_lang: str
    text: str
    source_characters: int


@dataclass(frozen=True)
class CostProjection:
    source_characters_per_second: float
    monthly_source_character_capacity: float
    gpu_cost_per_million_source_characters_cny: float


@dataclass(frozen=True)
class Observation:
    latency_seconds: float
    source_characters: int
    source_tokens: int
    translation: str | None
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
    source_characters: int
    source_characters_per_second: float
    source_tokens: int
    source_tokens_per_second: float
    output_tokens: int
    output_tokens_per_second: float
    latency_min_seconds: float
    latency_mean_seconds: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    latency_max_seconds: float
    error_categories: dict[str, int]
    monthly_source_character_capacity: float | None
    gpu_cost_per_million_source_characters_cny: float | None
    slo_passed: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    corpus_path: Path
    tokenizer: str
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
class CorpusSummary:
    record_count: int
    source_characters: int
    language_pairs: dict[str, int]


@dataclass(frozen=True)
class BenchmarkReport:
    tokenizer: str
    duration_seconds: float
    max_p95_seconds: float
    max_error_rate: float
    a10_monthly_cost_cny: float
    corpus: CorpusSummary
    levels: tuple[LevelResult, ...]
    selected_sustainable: LevelResult | None


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BenchmarkError(message)


def load_corpus(path: Path) -> list[CorpusRecord]:
    records = []
    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchmarkError(f"line {line_number}: invalid JSON") from error

            if not isinstance(value, dict):
                raise BenchmarkError(f"line {line_number}: record must be an object")

            unsupported_fields = value.keys() - _CORPUS_FIELDS
            if unsupported_fields:
                raise BenchmarkError(f"line {line_number}: unsupported fields")

            for field in ("source_lang", "target_lang", "text"):
                if field not in value:
                    raise BenchmarkError(f"line {line_number}: missing {field}")
                if not isinstance(value[field], str):
                    raise BenchmarkError(f"line {line_number}: {field} must be a string")
                if not value[field].strip():
                    raise BenchmarkError(f"line {line_number}: {field} must not be blank")

            records.append(
                CorpusRecord(
                    source_lang=value["source_lang"],
                    target_lang=value["target_lang"],
                    text=value["text"],
                    source_characters=len(value["text"]),
                )
            )

    if not records:
        raise BenchmarkError("corpus has no records")
    return records


def nearest_rank(values: Sequence[float], percentile: int) -> float:
    if not values:
        raise BenchmarkError("values must not be empty")
    if not 1 <= percentile <= 100:
        raise BenchmarkError("percentile must be between 1 and 100")

    sorted_values = sorted(values)
    rank = math.ceil(percentile / 100 * len(sorted_values))
    return sorted_values[rank - 1]


def project_gpu_cost(
    source_characters: float,
    elapsed_seconds: float,
    monthly_gpu_cost_cny: float,
) -> CostProjection:
    inputs = (source_characters, elapsed_seconds, monthly_gpu_cost_cny)
    if not all(math.isfinite(value) and value > 0 for value in inputs):
        raise BenchmarkError("cost projection inputs must be positive and finite")

    source_characters_per_second = source_characters / elapsed_seconds
    monthly_source_character_capacity = source_characters_per_second * MONTH_SECONDS
    gpu_cost_per_million_source_characters_cny = (
        monthly_gpu_cost_cny / monthly_source_character_capacity * 1_000_000
    )
    return CostProjection(
        source_characters_per_second=source_characters_per_second,
        monthly_source_character_capacity=monthly_source_character_capacity,
        gpu_cost_per_million_source_characters_cny=gpu_cost_per_million_source_characters_cny,
    )


async def send_translation_request(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    record: CorpusRecord,
    source_tokens: int,
) -> Observation:
    started_at = time.perf_counter()
    error_category = None
    translation = None
    try:
        response = await client.post(
            endpoint,
            headers={"X-API-Key": api_key},
            json={
                "source_lang": record.source_lang,
                "target_lang": record.target_lang,
                "text": record.text,
            },
        )
        if response.status_code != 200:
            error_category = f"http_{response.status_code}"
        else:
            try:
                body = response.json()
            except ValueError:
                error_category = "invalid_json"
            else:
                value = body.get("translation") if isinstance(body, dict) else None
                if isinstance(value, str):
                    translation = value
                else:
                    error_category = "invalid_response"
    except httpx.TimeoutException:
        error_category = "timeout"
    except httpx.RequestError:
        error_category = "connection_error"

    latency = time.perf_counter() - started_at
    succeeded = error_category is None
    return Observation(
        latency_seconds=latency,
        source_characters=record.source_characters if succeeded else 0,
        source_tokens=source_tokens if succeeded else 0,
        translation=translation,
        error_category=error_category,
    )


async def warm_up(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    records: Sequence[CorpusRecord],
    source_token_counts: Sequence[int],
    request_count: int,
) -> None:
    for index in range(request_count):
        corpus_index = index % len(records)
        observation = await send_translation_request(
            client,
            endpoint,
            api_key,
            records[corpus_index],
            source_token_counts[corpus_index],
        )
        if observation.error_category is not None:
            raise BenchmarkError(f"warmup failed: {observation.error_category}")


async def run_level(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    records: Sequence[CorpusRecord],
    source_token_counts: Sequence[int],
    concurrency: int,
    duration_seconds: float,
) -> tuple[list[Observation], float]:
    if concurrency <= 0 or duration_seconds <= 0:
        raise BenchmarkError("concurrency and duration must be positive")
    if not records or len(records) != len(source_token_counts):
        raise BenchmarkError("records and source token counts must align")

    next_index = 0
    start_event = asyncio.Event()
    started_at = time.perf_counter()
    deadline = started_at + duration_seconds

    async def worker() -> list[Observation]:
        nonlocal next_index
        worker_observations = []
        await start_event.wait()
        while time.perf_counter() < deadline:
            index = next_index % len(records)
            next_index += 1
            worker_observations.append(
                await send_translation_request(
                    client,
                    endpoint,
                    api_key,
                    records[index],
                    source_token_counts[index],
                )
            )
        return worker_observations

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    start_event.set()
    worker_results = await asyncio.gather(*tasks)
    measured_seconds = time.perf_counter() - started_at
    observations = [item for result in worker_results for item in result]
    return observations, measured_seconds


def aggregate_level(
    concurrency: int,
    measured_seconds: float,
    observations: Sequence[Observation],
    output_token_counts: Sequence[int],
    max_p95_seconds: float,
    max_error_rate: float,
    a10_monthly_cost_cny: float,
) -> LevelResult:
    if measured_seconds <= 0:
        raise BenchmarkError("measured seconds must be positive")
    successful = [item for item in observations if item.error_category is None]
    if len(output_token_counts) != len(successful):
        raise BenchmarkError("output token counts must match successful observations")

    attempted_count = len(observations)
    successful_count = len(successful)
    failed_count = attempted_count - successful_count
    error_rate = failed_count / attempted_count if attempted_count else 1.0
    latencies = [item.latency_seconds for item in observations]
    source_characters = sum(item.source_characters for item in successful)
    source_tokens = sum(item.source_tokens for item in successful)
    output_tokens = sum(output_token_counts)
    errors = Counter(
        item.error_category for item in observations if item.error_category is not None
    )

    if latencies:
        latency_min = min(latencies)
        latency_mean = statistics.fmean(latencies)
        latency_p50 = nearest_rank(latencies, 50)
        latency_p95 = nearest_rank(latencies, 95)
        latency_p99 = nearest_rank(latencies, 99)
        latency_max = max(latencies)
    else:
        latency_min = latency_mean = latency_p50 = latency_p95 = latency_p99 = latency_max = 0.0

    projection = (
        project_gpu_cost(source_characters, measured_seconds, a10_monthly_cost_cny)
        if source_characters
        else None
    )
    slo_passed = bool(
        attempted_count
        and successful_count
        and latency_p95 <= max_p95_seconds
        and error_rate <= max_error_rate
    )
    return LevelResult(
        concurrency=concurrency,
        measured_seconds=measured_seconds,
        attempted_requests=attempted_count,
        successful_requests=successful_count,
        failed_requests=failed_count,
        error_rate=error_rate,
        requests_per_second=successful_count / measured_seconds,
        source_characters=source_characters,
        source_characters_per_second=source_characters / measured_seconds,
        source_tokens=source_tokens,
        source_tokens_per_second=source_tokens / measured_seconds,
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens / measured_seconds,
        latency_min_seconds=latency_min,
        latency_mean_seconds=latency_mean,
        latency_p50_seconds=latency_p50,
        latency_p95_seconds=latency_p95,
        latency_p99_seconds=latency_p99,
        latency_max_seconds=latency_max,
        error_categories=dict(sorted(errors.items())),
        monthly_source_character_capacity=(
            projection.monthly_source_character_capacity if projection else None
        ),
        gpu_cost_per_million_source_characters_cny=(
            projection.gpu_cost_per_million_source_characters_cny if projection else None
        ),
        slo_passed=slo_passed,
    )


def select_sustainable_level(
    levels: Sequence[LevelResult],
) -> LevelResult | None:
    passing = [level for level in levels if level.slo_passed]
    return max(passing, key=lambda level: level.concurrency) if passing else None


def _positive_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def _error_rate(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _concurrency_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrency must contain integers") from exc
    if not levels or any(level <= 0 for level in levels) or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("concurrency levels must be unique positive integers")
    return levels


def parse_config(
    argv: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> BenchmarkConfig:
    environment = os.environ if environ is None else environ
    parser = _ArgumentParser(description="Benchmark the HY-MT HTTP translation service")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--concurrency", default=(1, 2, 4, 8, 16, 32), type=_concurrency_levels)
    parser.add_argument("--duration-seconds", default=30.0, type=_positive_finite)
    parser.add_argument("--warmup-requests", default=3, type=int)
    parser.add_argument("--request-timeout-seconds", default=30.0, type=_positive_finite)
    parser.add_argument("--max-p95-seconds", default=1.0, type=_positive_finite)
    parser.add_argument("--max-error-rate", default=0.001, type=_error_rate)
    parser.add_argument("--a10-monthly-cost-cny", default=2132.72, type=_positive_finite)
    args = parser.parse_args(argv)
    if args.warmup_requests < 0:
        raise BenchmarkError("warmup requests must not be negative")
    api_key = args.api_key or environment.get("API_KEY")
    if not api_key:
        raise BenchmarkError("API key is required")
    endpoint = environment.get(
        "MT_BENCHMARK_URL", "http://127.0.0.1:8000/v1/translate"
    )
    if not endpoint.strip():
        raise BenchmarkError("benchmark endpoint is required")
    return BenchmarkConfig(
        corpus_path=args.corpus,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        endpoint=endpoint,
        api_key=api_key,
        concurrency_levels=args.concurrency,
        duration_seconds=args.duration_seconds,
        warmup_requests=args.warmup_requests,
        request_timeout_seconds=args.request_timeout_seconds,
        max_p95_seconds=args.max_p95_seconds,
        max_error_rate=args.max_error_rate,
        a10_monthly_cost_cny=args.a10_monthly_cost_cny,
    )


def summarize_corpus(records: Sequence[CorpusRecord]) -> CorpusSummary:
    pairs = Counter(f"{record.source_lang}->{record.target_lang}" for record in records)
    return CorpusSummary(
        record_count=len(records),
        source_characters=sum(record.source_characters for record in records),
        language_pairs=dict(sorted(pairs.items())),
    )


def _report_payload(report: BenchmarkReport) -> dict[str, object]:
    return {
        "configuration": {
            "tokenizer": report.tokenizer,
            "duration_seconds": report.duration_seconds,
            "max_p95_seconds": report.max_p95_seconds,
            "max_error_rate": report.max_error_rate,
            "a10_monthly_cost_cny": report.a10_monthly_cost_cny,
            "month_seconds": MONTH_SECONDS,
        },
        "corpus": asdict(report.corpus),
        "levels": [asdict(level) for level in report.levels],
        "selected_sustainable": (
            asdict(report.selected_sustainable)
            if report.selected_sustainable is not None
            else None
        ),
    }


def render_json(report: BenchmarkReport) -> str:
    return json.dumps(_report_payload(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _display(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# MT 容量与成本压测报告",
        "",
        "本报告仅计算独立单张 A10 的 GPU 成本，不包含 CPU、内存、存储、网络、负载均衡、冗余和运维。",
        "Mock 后端或与 ASR/TTS 共用 GPU 的结果不能作为商业容量证据。",
        "",
        f"- 语料记录数：{report.corpus.record_count}",
        f"- 语料源字符数：{report.corpus.source_characters}",
        f"- A10 月成本：{report.a10_monthly_cost_cny:.2f} 元",
        f"- SLO：P95 ≤ {report.max_p95_seconds:.3f} 秒，错误率 ≤ {report.max_error_rate:.4%}",
        "",
        "## 各并发档结果",
        "",
        "| 并发 | 成功/请求 | RPS | 源字符/秒 | 输入 Token/秒 | 输出 Token/秒 | P50/P95/P99 秒 | 错误率 | 每百万源字符 GPU 成本 | SLO |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | :---: |",
    ]
    for level in report.levels:
        lines.append(
            "| "
            f"{level.concurrency} | {level.successful_requests}/{level.attempted_requests} | "
            f"{level.requests_per_second:.3f} | {level.source_characters_per_second:.3f} | "
            f"{level.source_tokens_per_second:.3f} | {level.output_tokens_per_second:.3f} | "
            f"{level.latency_p50_seconds:.3f}/{level.latency_p95_seconds:.3f}/{level.latency_p99_seconds:.3f} | "
            f"{level.error_rate:.4%} | {_display(level.gpu_cost_per_million_source_characters_cny, 4)} | "
            f"{'通过' if level.slo_passed else '失败'} |"
        )
    lines.extend(["", "## 可持续容量", ""])
    if report.selected_sustainable is None:
        lines.append("没有并发档同时满足延迟与错误率门槛。")
    else:
        selected = report.selected_sustainable
        lines.extend(
            [
                f"最高通过并发档：**{selected.concurrency}**。",
                f"每百万源字符 GPU 成本：**{_display(selected.gpu_cost_per_million_source_characters_cny, 4)} 元**。",
            ]
        )
    lines.extend(
        [
            "",
            "## 计算公式",
            "",
            "`月处理源字符 = 源字符/秒 × 2,592,000`",
            "",
            "`每百万源字符 GPU 成本 = A10 月成本 × 1,000,000 ÷ 月处理源字符`",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(output_dir: Path, report: BenchmarkReport) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mt-benchmark.json").write_text(render_json(report), encoding="utf-8")
    (output_dir / "mt-benchmark.md").write_text(render_markdown(report), encoding="utf-8")


def count_tokens(tokenizer: object, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


async def run_benchmark(
    config: BenchmarkConfig,
    records: Sequence[CorpusRecord],
    tokenizer: object,
) -> BenchmarkReport:
    source_token_counts = [count_tokens(tokenizer, record.text) for record in records]
    levels = []
    async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
        await warm_up(
            client,
            config.endpoint,
            config.api_key,
            records,
            source_token_counts,
            config.warmup_requests,
        )
        for concurrency in config.concurrency_levels:
            observations, measured_seconds = await run_level(
                client,
                config.endpoint,
                config.api_key,
                records,
                source_token_counts,
                concurrency,
                config.duration_seconds,
            )
            output_counts = [
                count_tokens(tokenizer, item.translation)
                for item in observations
                if item.translation is not None
            ]
            levels.append(
                aggregate_level(
                    concurrency,
                    measured_seconds,
                    observations,
                    output_counts,
                    config.max_p95_seconds,
                    config.max_error_rate,
                    config.a10_monthly_cost_cny,
                )
            )
    level_tuple = tuple(levels)
    return BenchmarkReport(
        tokenizer=config.tokenizer,
        duration_seconds=config.duration_seconds,
        max_p95_seconds=config.max_p95_seconds,
        max_error_rate=config.max_error_rate,
        a10_monthly_cost_cny=config.a10_monthly_cost_cny,
        corpus=summarize_corpus(records),
        levels=level_tuple,
        selected_sustainable=select_sustainable_level(level_tuple),
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_config(argv)
        records = load_corpus(config.corpus_path)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        report = asyncio.run(run_benchmark(config, records, tokenizer))
        write_reports(config.output_dir, report)
    except BenchmarkError as exc:
        print(f"MT benchmark failed: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError):
        print("MT benchmark failed: local configuration or artifact error", file=sys.stderr)
        return 2
    print("MT benchmark reports written: mt-benchmark.json, mt-benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
