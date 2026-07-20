import asyncio
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
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
