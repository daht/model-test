# MT Capacity and Cost Benchmark Design

## Objective

Add a repeatable command-line benchmark for the deployed HY-MT translation API. The benchmark must send real HTTP requests to the local service, identify the highest tested concurrency that meets the configured latency and error-rate limits, and calculate the single-A10 GPU cost per one million source Unicode characters.

The benchmark is a capacity and GPU-cost tool. It does not change translation serving behavior, optimize model inference, or claim that local mock results represent A10 capacity.

## Confirmed Commercial Accounting

- One billable source character is one Unicode code point from the request `text` value.
- Chinese characters, Latin characters, punctuation, spaces, and other Unicode code points all count.
- Cost is reported per 1,000,000 source characters.
- The default A10 price is CNY 2,132.72 per month, derived from CNY 25,592.64 per year.
- A month is 30 days of continuous operation: 2,592,000 seconds.
- The primary result is GPU-only cost. CPU, memory, storage, networking, load balancing, redundancy, and operations are explicitly excluded.
- The benchmark targets one independent A10. The translation service must not share that GPU with ASR or TTS during an accepted capacity run.

For an accepted concurrency result:

```text
source_characters_per_second = successful_source_characters / measured_seconds
monthly_source_character_capacity = source_characters_per_second * 2,592,000
gpu_cost_per_million_source_characters_cny =
    2,132.72 * 1,000,000 / monthly_source_character_capacity
```

Failed requests contribute to attempted request and error-rate metrics, but contribute no successful characters or tokens to capacity and cost.

## Input Corpus

The command reads UTF-8 JSON Lines. Each non-empty line must contain exactly the fields needed by the existing API:

```json
{"source_lang":"zh","target_lang":"en","text":"需要翻译的真实业务文本"}
```

`source_lang`, `target_lang`, and `text` must be non-empty strings. `glossary` and `preserve_format` are intentionally outside the first version because the confirmed benchmark workload only requires the three fields above. Invalid JSON, missing fields, unsupported extra value types, or blank values fail before any request is sent. Empty lines are ignored.

The corpus is loaded before measurement. Requests cycle through its records in stable file order so each concurrency level sees the same reproducible workload mix. A worker sends its next request only after its previous request completes, which models independent users making synchronous HTTP requests and produces a closed-loop concurrency benchmark.

## Command-Line Interface

Create `scripts/benchmark_mt.py` with these inputs:

- `--corpus`: required JSONL path.
- `--api-key`: optional; otherwise read `API_KEY` from the environment. The key is never printed or written to reports.
- `--url`: defaults to `http://127.0.0.1:8000/v1/translate`.
- `--tokenizer`: required model ID or local tokenizer path used before and after measurement for source and translated-text token counts.
- `--concurrency`: comma-separated positive integers, default `1,2,4,8,16,32`.
- `--duration-seconds`: measured duration for each level, default 30 seconds.
- `--warmup-requests`: excluded warmup request count before the first level, default 3.
- `--request-timeout-seconds`: default 30 seconds.
- `--max-p95-seconds`: default 1.0 second.
- `--max-error-rate`: default 0.001, representing 0.1%.
- `--a10-monthly-cost-cny`: default 2132.72.
- `--output-dir`: required directory for final reports.

The process exits nonzero for configuration, corpus, tokenizer, authentication, connectivity, or report-writing failures. A completed benchmark exits zero even when no tested concurrency satisfies the SLO; the reports make that result explicit.

## Measurement Flow

1. Validate all arguments and load the corpus.
2. Load the tokenizer and precompute source-text token counts outside the timed sections.
3. Create one reusable asynchronous HTTP client with the configured timeout and API key.
4. Send the excluded warmup requests sequentially. Any warmup failure stops the run because the service is not ready for capacity measurement.
5. For each concurrency level, start that many workers behind a common start event.
6. Each worker repeatedly selects the next corpus record, records monotonic start time, performs the real HTTP POST, records completion time, and stores a bounded per-request observation until the shared duration deadline.
7. A request succeeds only when the response status is 200 and the JSON response contains a string `translation`.
8. Tokenize successful translations after the timed level completes so local tokenizer CPU work does not distort HTTP latency or request throughput.
9. Aggregate the level, then continue to the next configured level using the same client.
10. Write reports only after all levels finish.

Requests already in flight when the duration deadline is reached are allowed to finish and are included. The measured wall-clock interval therefore ends when all workers finish their last in-flight request. This prevents cancelled requests from artificially improving latency and error metrics.

## Metrics and Sustainable Capacity

Each concurrency result contains:

- attempted, successful, and failed requests;
- error rate (`failed / attempted`);
- measured wall-clock seconds;
- requests per second based on successful requests;
- successful source characters and source characters per second;
- successful source tokens and source tokens per second;
- output tokens and output tokens per second;
- latency minimum, mean, P50, P95, P99, and maximum;
- HTTP status/error category counts;
- projected monthly source-character capacity;
- GPU cost per million source characters in CNY;
- whether both configured SLO thresholds passed.

Percentiles use a deterministic nearest-rank calculation. A level with no attempted requests, no successful requests, or no latency observations cannot pass.

The selected sustainable result is the highest configured concurrency level that satisfies both:

```text
P95 latency <= max_p95_seconds
error rate <= max_error_rate
```

The report does not assume that passing levels are monotonic. It records every level and selects the highest passing configured value.

## Reports

Write stable files under `--output-dir`:

- `mt-benchmark.json`: complete configuration without the API key, corpus summary, per-level measurements, and selected sustainable result.
- `mt-benchmark.md`: human-readable methodology, exclusions, results table, selected result, formulas, and a warning that mock backends or shared GPUs cannot establish commercial A10 capacity.

Reports include the endpoint URL but never include request/response text, translations, or credentials. This keeps business content and secrets out of benchmark artifacts.

## Error Handling

- Validate all local inputs before opening the HTTP client.
- Fail immediately on an absent API key without displaying its value.
- Fail warmup on non-200 responses, invalid JSON, or a missing translation.
- During measurement, record request timeouts, connection failures, HTTP statuses, invalid JSON, and invalid response schemas as categorized failures, then continue until the level ends.
- Fail the whole command if the tokenizer cannot load or reports cannot be written.
- Do not retry requests. Retries would hide overload and inflate character processing counts.

## Code Structure

- `scripts/benchmark_mt.py`: CLI parsing, validated corpus loading, asynchronous HTTP workload, deterministic aggregation, cost calculation, and JSON/Markdown rendering. These functions remain importable for focused tests.
- `tests/test_benchmark_mt.py`: unit and in-process HTTP tests for validation, metrics, cost, workload behavior, failure accounting, credential redaction, and report rendering.
- `docs/mt-capacity-cost-benchmark.md`: operator instructions, JSONL example, local invocation, metric definitions, commercial formula, and interpretation rules.

No production API, schema, model wrapper, deployment configuration, or unrelated ASR file changes are required.

## Test Strategy

Development follows test-first red-green cycles:

1. Corpus validation and exact Unicode character counting.
2. Deterministic percentile, throughput, monthly capacity, and cost calculations.
3. SLO pass/fail and highest sustainable level selection.
4. Async HTTP workload against an in-process ASGI test application, proving real HTTP request construction, concurrency, successful translation parsing, and failure categorization without a GPU.
5. JSON and Markdown report content and API-key redaction.
6. CLI argument and environment validation.

The focused test file and the full repository pytest suite must pass locally. Python compilation and `git diff --check` complete the local verification. Real A10 capacity remains an external benchmark and must be reported as unavailable until the command is run against the warmed production model on an independent A10.

## Acceptance Criteria

- A user can point the CLI at the local `/v1/translate` HTTP endpoint and a real JSONL corpus.
- Every request goes through the deployed HTTP application rather than importing the translator.
- Reports reproduce the confirmed per-million-source-character A10 GPU cost formula.
- Reports provide enough latency, throughput, token, character, and error evidence to choose a sustainable concurrency level.
- Credentials and corpus text never appear in output artifacts or normal terminal output.
- No existing API behavior changes.
