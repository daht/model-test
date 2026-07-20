import json
import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Header

from scripts.benchmark_mt import (
    BenchmarkError,
    BenchmarkReport,
    CorpusRecord,
    CorpusSummary,
    Observation,
    aggregate_level,
    load_corpus,
    nearest_rank,
    project_gpu_cost,
    parse_config,
    render_json,
    render_markdown,
    run_level,
    select_sustainable_level,
)


def write_jsonl(tmp_path, lines):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    return corpus_path


def test_load_corpus_preserves_utf8_text_and_counts_all_characters(tmp_path):
    corpus_path = write_jsonl(
        tmp_path,
        [json.dumps({"source_lang": "zh", "target_lang": "en", "text": "你好, A B！"}, ensure_ascii=False)],
    )

    assert load_corpus(corpus_path) == [CorpusRecord("zh", "en", "你好, A B！", 8)]


def test_load_corpus_ignores_empty_lines(tmp_path):
    corpus_path = write_jsonl(
        tmp_path,
        ["", '{"source_lang":"zh","target_lang":"en","text":"你好"}', "   "],
    )

    assert load_corpus(corpus_path) == [CorpusRecord("zh", "en", "你好", 2)]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["", "   "], "corpus has no records"),
        (['{"source_lang":"zh","target_lang":"en"}'], "line 1: missing text"),
        (['{"source_lang":"zh","target_lang":"en","text":"  "}'], "line 1: text must not be blank"),
        (
            ['{"source_lang":"zh","target_lang":"en","text":"你好","extra":true}'],
            "line 1: unsupported fields",
        ),
        (["[]"], "line 1: record must be an object"),
        (['{"source_lang":1,"target_lang":"en","text":"你好"}'], "line 1: source_lang must be a string"),
        (['{"source_lang":" ","target_lang":"en","text":"你好"}'], "line 1: source_lang must not be blank"),
        (['{"source_lang":"zh","target_lang":1,"text":"你好"}'], "line 1: target_lang must be a string"),
        (['{"source_lang":"zh","target_lang":" ","text":"你好"}'], "line 1: target_lang must not be blank"),
        (['{"source_lang":"zh","target_lang":"en","text":1}'], "line 1: text must be a string"),
        (["not-json"], "line 1: invalid JSON"),
    ],
)
def test_load_corpus_rejects_invalid_records(tmp_path, lines, message):
    corpus_path = write_jsonl(tmp_path, lines)

    with pytest.raises(BenchmarkError, match=message):
        load_corpus(corpus_path)


def test_nearest_rank_uses_ceiling_rank():
    values = [0.1, 0.2, 0.3, 0.4]

    assert nearest_rank(values, 50) == 0.2
    assert nearest_rank(values, 95) == 0.4


@pytest.mark.parametrize("values", [[], ()])
def test_nearest_rank_rejects_empty_values(values):
    with pytest.raises(BenchmarkError):
        nearest_rank(values, 50)


@pytest.mark.parametrize("percentile", [0, 101])
def test_nearest_rank_rejects_percentiles_outside_one_to_one_hundred(percentile):
    with pytest.raises(BenchmarkError):
        nearest_rank([1.0], percentile)


def test_project_gpu_cost_calculates_throughput_capacity_and_unit_cost():
    projection = project_gpu_cost(2_592_000, 2_592_000, 2132.72)

    assert projection.source_characters_per_second == pytest.approx(1.0)
    assert projection.monthly_source_character_capacity == pytest.approx(2_592_000)
    assert projection.gpu_cost_per_million_source_characters_cny == pytest.approx(822.8086)


@pytest.mark.parametrize(
    "source_characters, elapsed_seconds, monthly_gpu_cost_cny",
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (float("nan"), 1, 1),
        (1, float("nan"), 1),
        (1, 1, float("nan")),
        (float("inf"), 1, 1),
        (1, float("inf"), 1),
        (1, 1, float("inf")),
    ],
)
def test_project_gpu_cost_rejects_non_positive_inputs(
    source_characters, elapsed_seconds, monthly_gpu_cost_cny
):
    with pytest.raises(BenchmarkError):
        project_gpu_cost(source_characters, elapsed_seconds, monthly_gpu_cost_cny)


def test_aggregate_level_calculates_capacity_cost_and_slo():
    observations = [
        Observation(0.2, 10, 4, "translated", None),
        Observation(0.4, 20, 8, "more words", None),
        Observation(0.5, 0, 0, None, "http_503"),
    ]

    result = aggregate_level(
        concurrency=4,
        measured_seconds=2.0,
        observations=observations,
        output_token_counts=[3, 5],
        max_p95_seconds=1.0,
        max_error_rate=0.5,
        a10_monthly_cost_cny=2132.72,
    )

    assert (result.attempted_requests, result.successful_requests, result.failed_requests) == (3, 2, 1)
    assert result.requests_per_second == 1.0
    assert result.source_characters_per_second == 15.0
    assert result.output_tokens_per_second == 4.0
    assert result.error_categories == {"http_503": 1}
    assert result.slo_passed is True


def test_select_sustainable_level_uses_highest_passing_level():
    passing_one = aggregate_level(1, 1.0, [Observation(0.1, 1, 1, "x", None)], [1], 1.0, 0.0, 1.0)
    passing_four = aggregate_level(4, 1.0, [Observation(0.1, 1, 1, "x", None)], [1], 1.0, 0.0, 1.0)
    failing_eight = aggregate_level(8, 1.0, [Observation(2.0, 1, 1, "x", None)], [1], 1.0, 0.0, 1.0)

    assert select_sustainable_level([passing_one, failing_eight, passing_four]) == passing_four


def test_run_level_uses_real_concurrent_http_requests():
    asyncio.run(_assert_real_concurrent_http_requests())


async def _assert_real_concurrent_http_requests():
    app = FastAPI()
    active = 0
    maximum_active = 0
    both_arrived = asyncio.Event()
    payloads = []

    @app.post("/v1/translate")
    async def translate(payload: dict, x_api_key: str = Header()):
        nonlocal active, maximum_active
        assert x_api_key == "secret"
        payloads.append(payload)
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=1)
        active -= 1
        return {"translation": "ok"}

    records = [CorpusRecord("zh", "en", "你好", 2)]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as client:
        observations, _ = await run_level(
            client, "http://test/v1/translate", "secret", records, [1], 2, 0.05
        )

    assert maximum_active == 2
    assert len(observations) >= 2
    assert all(item.error_category is None for item in observations)
    assert payloads[0] == {"source_lang": "zh", "target_lang": "en", "text": "你好"}


def test_parse_config_uses_approved_defaults_and_environment(tmp_path):
    config = parse_config(
        ["--corpus", str(tmp_path / "input.jsonl"), "--tokenizer", "model", "--output-dir", str(tmp_path)],
        {"API_KEY": "secret", "MT_BENCHMARK_URL": "http://sensitive.invalid/v1/translate"},
    )

    assert config.concurrency_levels == (1, 2, 4, 8, 16, 32)
    assert config.duration_seconds == 30.0
    assert config.max_p95_seconds == 1.0
    assert config.max_error_rate == 0.001
    assert config.a10_monthly_cost_cny == 2132.72
    assert config.endpoint == "http://sensitive.invalid/v1/translate"
    assert config.api_key == "secret"


@pytest.mark.parametrize("value", ["0", "1,1", "a", "1,-2"])
def test_parse_config_rejects_invalid_concurrency(tmp_path, value):
    with pytest.raises(BenchmarkError):
        parse_config(
            [
                "--corpus", str(tmp_path / "input.jsonl"),
                "--tokenizer", "model",
                "--output-dir", str(tmp_path),
                "--concurrency", value,
            ],
            {"API_KEY": "secret"},
        )


def test_parse_config_requires_api_key(tmp_path):
    with pytest.raises(BenchmarkError, match="API key is required"):
        parse_config(
            ["--corpus", str(tmp_path / "input.jsonl"), "--tokenizer", "model", "--output-dir", str(tmp_path)],
            {},
        )


def test_reports_include_metrics_and_exclude_sensitive_values():
    level = aggregate_level(
        4,
        2.0,
        [Observation(0.2, 10, 4, "translated-secret", None)],
        [3],
        1.0,
        0.001,
        2132.72,
    )
    report = BenchmarkReport(
        tokenizer="model",
        duration_seconds=30.0,
        max_p95_seconds=1.0,
        max_error_rate=0.001,
        a10_monthly_cost_cny=2132.72,
        corpus=CorpusSummary(1, 10, {"zh->en": 1}),
        levels=(level,),
        selected_sustainable=level,
    )

    json_text = render_json(report)
    markdown = render_markdown(report)
    combined = json_text + markdown

    assert "gpu_cost_per_million_source_characters_cny" in json_text
    assert "每百万源字符 GPU 成本" in markdown
    assert "http://sensitive.invalid" not in combined
    assert "api-key-secret" not in combined
    assert "translated-secret" not in combined
