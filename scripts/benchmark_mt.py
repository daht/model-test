import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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
    if not (
        source_characters > 0 and elapsed_seconds > 0 and monthly_gpu_cost_cny > 0
    ):
        raise BenchmarkError("cost projection inputs must be positive")

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
