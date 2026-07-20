# MT Capacity and Cost Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI that load-tests the real HY-MT HTTP API and reports sustainable single-A10 throughput plus GPU cost per one million source Unicode characters.

**Architecture:** One importable Python script owns strict JSONL loading, asynchronous closed-loop HTTP load, deterministic aggregation, cost projection, report rendering, and CLI orchestration. It calls the deployed API over HTTP and never imports the translator. Focused pytest tests drive each behavior before implementation.

**Tech Stack:** Python 3.12, asyncio, httpx, Transformers tokenizer, pytest, FastAPI ASGI transport, JSON, Markdown.

---

## File Structure

- Create `scripts/benchmark_mt.py`: benchmark types, validation, HTTP runner, metrics, reports, and CLI.
- Create `tests/test_benchmark_mt.py`: unit and ASGI-backed integration tests.
- Create `docs/mt-capacity-cost-benchmark.md`: safe operator runbook.
- Modify `README.md`: one runbook link in the file inventory.

### Task 1: Corpus and accounting primitives

**Files:**
- Create: `scripts/benchmark_mt.py`
- Create: `tests/test_benchmark_mt.py`

- [ ] **Step 1: Write failing corpus and formula tests**

Create an import helper for `scripts/benchmark_mt.py`, then add:

```python
def test_load_corpus_counts_all_unicode_code_points(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"source_lang":"zh","target_lang":"en","text":"你好, A B！"}\n',
        encoding="utf-8",
    )
    assert benchmark_mt.load_corpus(corpus) == [
        benchmark_mt.CorpusRecord("zh", "en", "你好, A B！", 8)
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "corpus has no records"),
        ('{"source_lang":"zh","target_lang":"en"}\n', "missing text"),
        ('{"source_lang":"zh","target_lang":"en","text":"   "}\n', "text must not be blank"),
        ('{"source_lang":"zh","target_lang":"en","text":"x","extra":1}\n', "unsupported fields"),
    ],
)
def test_load_corpus_rejects_invalid_records(tmp_path, content, message):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(content, encoding="utf-8")
    with pytest.raises(benchmark_mt.BenchmarkError, match=message):
        benchmark_mt.load_corpus(corpus)


def test_nearest_rank_percentile_is_deterministic():
    assert benchmark_mt.nearest_rank([0.1, 0.2, 0.3, 0.4], 50) == 0.2
    assert benchmark_mt.nearest_rank([0.1, 0.2, 0.3, 0.4], 95) == 0.4


def test_gpu_cost_uses_thirty_day_month():
    result = benchmark_mt.project_gpu_cost(2_592_000, 2_592_000, 2132.72)
    assert result.source_characters_per_second == 1.0
    assert result.monthly_source_character_capacity == 2_592_000.0
    assert result.gpu_cost_per_million_source_characters_cny == pytest.approx(822.8086)
```

- [ ] **Step 2: Run RED**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py`.

Expected: collection fails because the benchmark module is absent.

- [ ] **Step 3: Add minimal validated primitives**

Implement these public contracts:

```python
MONTH_SECONDS = 30 * 24 * 60 * 60
ALLOWED_CORPUS_FIELDS = {"source_lang", "target_lang", "text"}

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
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise BenchmarkError(f"line {line_number}: record must be an object")
        if set(value) - ALLOWED_CORPUS_FIELDS:
            raise BenchmarkError(f"line {line_number}: unsupported fields")
        for field in ("source_lang", "target_lang", "text"):
            if field not in value:
                raise BenchmarkError(f"line {line_number}: missing {field}")
            if not isinstance(value[field], str) or not value[field].strip():
                raise BenchmarkError(f"line {line_number}: {field} must not be blank")
        records.append(CorpusRecord(value["source_lang"], value["target_lang"], value["text"], len(value["text"])))
    if not records:
        raise BenchmarkError("corpus has no records")
    return records

def nearest_rank(values: Sequence[float], percentile: int) -> float:
    if not values or percentile < 1 or percentile > 100:
        raise BenchmarkError("invalid percentile input")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]

def project_gpu_cost(source_characters: int, measured_seconds: float, a10_monthly_cost_cny: float) -> CostProjection:
    if source_characters <= 0 or measured_seconds <= 0 or a10_monthly_cost_cny <= 0:
        raise BenchmarkError("cost projection requires positive inputs")
    per_second = source_characters / measured_seconds
    monthly = per_second * MONTH_SECONDS
    return CostProjection(per_second, monthly, a10_monthly_cost_cny * 1_000_000 / monthly)
```

`load_corpus` ignores empty lines, accepts exactly `source_lang`, `target_lang`, and `text`, rejects blank/non-string values with line numbers, and uses `len(text)` without normalization. `nearest_rank` sorts values and uses `ceil(p / 100 * n) - 1`. Cost uses the confirmed 2,592,000-second month and positive inputs only.

- [ ] **Step 4: Run GREEN**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py`; expect all Task 1 tests to pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_mt.py tests/test_benchmark_mt.py
git commit -m "feat(mt): add benchmark accounting primitives"
```

### Task 2: Real concurrent HTTP workload and aggregation

**Files:**
- Modify: `scripts/benchmark_mt.py`
- Modify: `tests/test_benchmark_mt.py`

- [ ] **Step 1: Write failing workload and aggregation tests**

Use `httpx.ASGITransport` with a FastAPI app and an `asyncio.Event` barrier. Run concurrency 2, assert two handlers overlap, and verify the API key header and exact corpus payload. Add cases proving HTTP 503, timeout/connection failure, invalid JSON, and missing `translation` are categorized and never retried.

Add deterministic aggregation:

```python
def test_aggregate_level_calculates_capacity_cost_and_slo():
    observations = [
        benchmark_mt.Observation(0.2, 10, 4, "translated", None),
        benchmark_mt.Observation(0.4, 20, 8, "more words", None),
        benchmark_mt.Observation(0.5, 0, 0, None, "http_503"),
    ]
    result = benchmark_mt.aggregate_level(
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
```

Add a selection test with passing levels 1 and 4 plus failing level 8; expect concurrency 4.

- [ ] **Step 2: Run RED**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py`; expect missing workload types/functions.

- [ ] **Step 3: Implement minimal HTTP execution**

Add `Observation` and `LevelResult` dataclasses. Implement `send_translation_request`, `warm_up`, `run_level`, `aggregate_level`, and `select_sustainable_level` with the exact parameter names exercised by the tests. The closed-loop core is:

```python
next_index = 0
start_event = asyncio.Event()
started_at = time.perf_counter()
deadline = started_at + duration_seconds

async def worker() -> list[Observation]:
    nonlocal next_index
    observations = []
    await start_event.wait()
    while time.perf_counter() < deadline:
        index = next_index % len(records)
        next_index += 1
        observations.append(
            await send_translation_request(
                client,
                endpoint,
                api_key,
                records[index],
                source_token_counts[index],
            )
        )
    return observations

tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
start_event.set()
worker_results = await asyncio.gather(*tasks)
elapsed = time.perf_counter() - started_at
observations = [item for worker_result in worker_results for item in worker_result]
```

Use monotonic timing, one shared async client, a common worker start event, stable cyclic corpus order, and closed-loop workers. Let final in-flight requests finish. Never retry. Categories are exactly `timeout`, `connection_error`, `http_<status>`, `invalid_json`, and `invalid_response`. Capacity numerators include successful requests only; latency includes all attempts. SLO requires observations, a success, `p95 <= limit`, and `error_rate <= limit`.

- [ ] **Step 4: Run GREEN**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py`; expect all Task 1-2 tests to pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_mt.py tests/test_benchmark_mt.py
git commit -m "feat(mt): benchmark concurrent HTTP translation"
```

### Task 3: Tokenizer, reports, and CLI

**Files:**
- Modify: `scripts/benchmark_mt.py`
- Modify: `tests/test_benchmark_mt.py`

- [ ] **Step 1: Write failing token, report, and CLI tests**

Use a fake tokenizer and assert source tokens are prepared before load measurement and successful output tokens are counted after a level. Render a fixed report and assert JSON and Markdown contain RPS, character/token throughput, P50/P95/P99, errors, projected monthly characters, cost per million, SLO state, and selected capacity.

Assert that endpoint, API key, corpus path/text, and translations never occur in either report or normal success output. Add parser/config tests for defaults and validation:

```python
assert config.concurrency_levels == (1, 2, 4, 8, 16, 32)
assert config.duration_seconds == 30.0
assert config.max_p95_seconds == 1.0
assert config.max_error_rate == 0.001
assert config.a10_monthly_cost_cny == 2132.72
```

Reject zero/duplicate/non-integer concurrency, nonpositive durations/timeouts/costs, error rates outside 0..1, absent API key, and unwritable output. Verify `MT_BENCHMARK_URL` overrides the localhost fallback but is not serializable.

- [ ] **Step 2: Run RED**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py`; expect missing tokenizer/report/CLI behaviors.

- [ ] **Step 3: Implement CLI orchestration**

Add `BenchmarkConfig`, `CorpusSummary`, and `BenchmarkReport`. Parse the approved CLI, resolve endpoint from `MT_BENCHMARK_URL`, key from `--api-key` or `API_KEY`, and load `AutoTokenizer` only inside `main`. Count with `tokenizer.encode(text, add_special_tokens=False)` outside timed HTTP intervals.

Implement `render_json`, `render_markdown`, and `write_reports` producing stable `mt-benchmark.json` and `mt-benchmark.md`. Renderer inputs must not contain endpoint, key, corpus path/text, or translations. JSON uses UTF-8, sorted keys, and indentation. Markdown states GPU-only exclusions, exact formula, SLO thresholds, per-level table, selected result, and mock/shared-GPU warning.

`main` returns nonzero for validated configuration/corpus/tokenizer/warmup/report failures, zero after a completed benchmark even with no passing level, and never stringifies connectivity exceptions that could reveal the endpoint.

- [ ] **Step 4: Run GREEN**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py`; expect all focused tests to pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_mt.py tests/test_benchmark_mt.py
git commit -m "feat(mt): report capacity and per-million cost"
```

### Task 4: Operator documentation and verification

**Files:**
- Create: `docs/mt-capacity-cost-benchmark.md`
- Modify: `README.md`
- Modify: `tests/test_benchmark_mt.py`

- [ ] **Step 1: Write a failing documentation contract test**

Read the runbook and require the script name, JSONL schema, `MT_BENCHMARK_URL`, `API_KEY`, monthly price, per-million formula, output filenames, and warnings about mock/shared GPUs. Reject public IPv4 literals and example secrets.

- [ ] **Step 2: Run RED**

Run `.venv/bin/pytest -q tests/test_benchmark_mt.py -k documentation`; expect missing runbook failure.

- [ ] **Step 3: Add the minimal runbook and README link**

Document prerequisites, safe environment configuration without actual values, JSONL preparation, a command containing no endpoint/key, metrics, formula, SLO interpretation, and the real independent-A10 gate. Add this inventory line:

```markdown
- `docs/mt-capacity-cost-benchmark.md`: real HTTP MT capacity and per-million-source-character GPU cost benchmark.
```

- [ ] **Step 4: Verify focused and full repository behavior**

Run:

```bash
.venv/bin/pytest -q tests/test_benchmark_mt.py
.venv/bin/pytest -q
.venv/bin/python -m py_compile scripts/benchmark_mt.py tests/test_benchmark_mt.py
git diff --check
git status --short
```

Expected: all tests pass, compilation and diff checks exit zero, and status lists only intended files.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/mt-capacity-cost-benchmark.md tests/test_benchmark_mt.py
git commit -m "docs(mt): add capacity benchmark runbook"
```

## Final Review and Handoff

- [ ] Run independent specification and code-quality reviews for every task, with correction and re-review loops.
- [ ] Dispatch a final read-only reviewer over the complete implementation range and resolve all Critical or Important findings.
- [ ] Re-run full verification after the final correction.
- [ ] Report SHAs and test evidence without endpoint/key values.
- [ ] Do not claim a commercial cost result before a warmed independent-A10 run.
