import json

import pytest

from scripts.benchmark_mt import (
    BenchmarkError,
    CorpusRecord,
    load_corpus,
    nearest_rank,
    project_gpu_cost,
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
