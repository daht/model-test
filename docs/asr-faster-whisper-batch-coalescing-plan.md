# Faster-Whisper Batch Coalescing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the A10 faster-whisper candidate coalesce offset compatible streams into bounded dynamic microbatches and expose enough aggregate buffer data to diagnose future lag safely.

**Architecture:** Keep the existing deadline-bounded `GatewayScheduler` and tune the production faster-whisper contract to a 200 ms collection window with six seconds of per-session PCM headroom. Prove the production file drives two 100 ms-offset sessions into one batch, then extend gateway metrics with current and lifetime buffer gauges without exposing session or transcript data.

**Tech Stack:** Python 3.12 tests, asyncio, FastAPI gateway, Pydantic settings, Bash release verifier, pytest, Docker Compose, CTranslate2/faster-whisper runtime.

---

## File Map

- Modify `cloud/A10.faster-whisper.env.example`: selected A10 coalescing and bounded-buffer values.
- Modify `scripts/verify_asr_release.sh`: backend-specific release enforcement for those values.
- Modify `cloud/README-A10.md`: operational rationale and capacity acceptance instructions.
- Modify `app/asr_gateway_metrics.py`: sanitized current and high-water session buffer gauges.
- Modify `app/asr_gateway.py`: feed per-session buffered/reserved accounting into metrics after ingest, cleanup, open, and release.
- Modify `tests/test_asr_deployment_scripts.py`: production template and release verifier contract.
- Modify `tests/test_asr_gateway.py`: production-config-driven offset-arrival batching regression.
- Modify `tests/test_asr_gateway_metrics.py`: metric behavior, reset, and secret-surface regression.
- Include `docs/asr-faster-whisper-batch-coalescing-design.md` and this plan in the accepted branch history.

## Task 1: Lock the production coalescing contract

**Files:**
- Modify: `tests/test_asr_deployment_scripts.py`
- Modify: `cloud/A10.faster-whisper.env.example`
- Modify: `scripts/verify_asr_release.sh`

- [x] **Step 1: Add failing deployment contract assertions**

Extend the existing required set in
`test_a10_faster_whisper_example_is_fp16_large_v3_batch_four_transcribe_only`
with these five complete entries:

```python
required = {
    "ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=200",
    "ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=6.0",
    "ASR_GATEWAY_DEFAULT_UPDATE_MS=2000",
    "ASR_MAX_CONNECTION_LAG_SECONDS=4.0",
    "ASR_MAX_UNDECODED_AGE_SECONDS=8.0",
}
```

Extend `test_release_gate_accepts_faster_whisper_contract_and_runs_gateway_warmup`:

```python
assert '"ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS": "200"' in verifier
assert '"ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS": "6.0"' in verifier
```

- [x] **Step 2: Run the two tests and verify RED**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_deployment_scripts.py::test_a10_faster_whisper_example_is_fp16_large_v3_batch_four_transcribe_only \
  tests/test_asr_deployment_scripts.py::test_release_gate_accepts_faster_whisper_contract_and_runs_gateway_warmup \
  -q -p no:cacheprovider
```

Expected: both tests fail because the candidate still contains `20` and `4.0`, and the release verifier does not enforce the new values.

- [x] **Step 3: Apply the minimal production configuration**

Change only the faster-whisper candidate:

```dotenv
ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=200
ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=6.0
```

In the `backend == "faster_whisper"` release requirements, add:

```python
"ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS": "200",
"ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS": "6.0",
```

Do not change global `Settings` defaults or the Qwen A10 example.

- [x] **Step 4: Run the two tests and verify GREEN**

Run the Step 2 command again.

Expected: `2 passed`.

## Task 2: Prove offset real-time sessions coalesce

**Files:**
- Modify: `tests/test_asr_gateway.py`

- [x] **Step 1: Add a production-config-driven deterministic regression**

Add a small dotenv reader that extracts non-secret ASR values from
`cloud/A10.faster-whisper.env.example`. Add a test with this structure:

```python
def test_faster_whisper_candidate_coalesces_offset_sessions_before_buffer_pressure():
    async def scenario():
        values = read_example_values("cloud/A10.faster-whisper.env.example")
        clock = FakeClock()
        adapter = FakeAdapter(dynamic=True)
        adapter.capabilities = replace(
            adapter.capabilities,
            streaming_mode=StreamingMode.ROLLING,
        )
        adapter.release.set()
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="rolling",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_max_active_sessions=2,
            asr_gateway_schedule_max_wait_ms=int(
                values["ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS"]
            ),
            asr_gateway_max_session_buffer_seconds=float(
                values["ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS"]
            ),
            asr_gateway_max_queued_audio_seconds=16,
            asr_gateway_default_update_ms=2000,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter}, clock=clock)
        await adapter.warmup()
        await runtime.registry.register(adapter.capabilities)
        first = await runtime.open_session("first", language="zh", options={})
        second = await runtime.open_session("second", language="zh", options={})
        frame = b"\x01\x00" * 32_000

        for _ in range(3):
            await runtime.ingest(first, frame, force=True)
            assert await runtime.scheduler.run_once("fake") == []
            clock.advance(0.1)
            assert await runtime.scheduler.run_once("fake") == []
            await runtime.ingest(second, frame, force=True)
            assert await runtime.scheduler.run_once("fake") == []
            clock.advance(0.101)
            results = await runtime.scheduler.run_once("fake")
            assert len(results) == 2

        snapshots = [first.sample_accounting, second.sample_accounting]
        await runtime.close()
        return [len(call) for call in adapter.calls], snapshots

    calls, snapshots = asyncio.run(scenario())
    assert calls == [2, 2, 2]
    assert all(item["buffered"] == item["reserved"] == 0 for item in snapshots)
```

Use a local deterministic `FakeClock` and a helper name that does not collide
with existing tests. Do not use `sleep`.

- [x] **Step 2: Verify sensitivity against the pre-fix production values**

Temporarily run the test with the candidate values read as `20` and `4.0` by
executing it against parent commit `266a6e5` in a temporary detached worktree,
or temporarily restore those two file lines without staging them.

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py::test_faster_whisper_candidate_coalesces_offset_sessions_before_buffer_pressure \
  -q -p no:cacheprovider
```

Expected pre-fix failure: the first job becomes due at 20 ms, so the 100 ms
offset produces a singleton instead of an empty pre-deadline result and later a
two-item batch.

- [x] **Step 3: Restore the selected production values and verify GREEN**

Run the same command with `200` and `6.0` restored.

Expected: `1 passed` with adapter call sizes `[2, 2, 2]` and zero buffered or
reserved samples after each accepted cycle.

## Task 3: Add sanitized buffer observability

**Files:**
- Modify: `tests/test_asr_gateway_metrics.py`
- Modify: `app/asr_gateway_metrics.py`
- Modify: `app/asr_gateway.py`
- Modify: `tests/test_asr_gateway.py`

- [x] **Step 1: Add failing metric unit tests**

Extend the metrics test to call:

```python
metrics.set_gauges(
    active_sessions=2,
    ready_depth=1,
    queued_samples=32_000,
    session_buffered_samples=32_000,
    session_reserved_samples=32_000,
    max_session_held_samples=64_000,
    sample_rate=16_000,
)
```

Assert:

```python
assert snapshot["session_buffered_audio_seconds"] == 2
assert snapshot["session_reserved_audio_seconds"] == 2
assert snapshot["max_session_held_audio_seconds"] == 4
assert snapshot["session_buffer_high_water_seconds"] == 4
```

Call `set_gauges` again with zero sample values and assert the three current
gauges return to zero while `session_buffer_high_water_seconds` remains four.
Keep the existing forbidden-word assertion.

- [x] **Step 2: Run the metric test and verify RED**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway_metrics.py::test_job_timeline_and_bounded_aggregates_are_sanitized \
  -q -p no:cacheprovider
```

Expected: failure because `GatewayMetrics.set_gauges` does not accept the new
sample values and the snapshot lacks the fields.

- [x] **Step 3: Implement current and high-water gauges**

Add a numeric `_session_buffer_high_water_samples` initialized to zero. Extend
`set_gauges` with required keyword-only sample arguments and update it with:

```python
self._session_buffer_high_water_samples = max(
    self._session_buffer_high_water_samples,
    max(0, max_session_held_samples),
)
```

Add the four seconds-valued keys to `_gauges`. Include the persistent high-water
key in both empty and completed snapshots. Preserve all existing metric names.

- [x] **Step 4: Feed session accounting into metrics**

In `GatewayRuntime._update_gauges`, derive accounting from every active context:

```python
accounting = [ctx.session.sample_accounting for ctx in self._contexts.values()]
buffered = sum(item["buffered"] for item in accounting)
reserved = sum(item["reserved"] for item in accounting)
max_held = max(
    (item["buffered"] + item["reserved"] for item in accounting),
    default=0,
)
```

Pass these values to `metrics.set_gauges`. In `ingest`, call `_update_gauges()`
after `_schedule_next` and before returning so queued/reserved pressure is
observable before cleanup.

- [x] **Step 5: Add and run a runtime gauge reset test**

Add a focused `tests/test_asr_gateway.py` test using a held `FakeAdapter`
submission. Assert the metrics snapshot reports a two-second reservation while
the job is held, then current values return to zero after release and cleanup,
while the high-water remains two seconds.

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway_metrics.py \
  tests/test_asr_gateway.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass.

## Task 4: Update the A10 runbook

**Files:**
- Modify: `cloud/README-A10.md`
- Modify: `tests/test_asr_deployment_scripts.py`

- [x] **Step 1: Add failing runbook assertions**

Require the runbook to state that 200 ms is bounded coalescing latency, six
seconds is jitter headroom rather than an accepted lag target, and the live
sweep must record session buffer high-water.

- [x] **Step 2: Run the focused deployment test and verify RED**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_deployment_scripts.py::test_runbook_has_faster_whisper_model_manifest_validation_and_qwen_rollback \
  -q -p no:cacheprovider
```

Expected: failure because the rationale is absent.

- [x] **Step 3: Add the concise operational explanation and verify GREEN**

Document the selected values, retained lag limits, metrics to record, and the
rule that capacity is not accepted until the first failing concurrency stage is
known. Run the Step 2 command again and expect `1 passed`.

## Task 5: Verify and commit the candidate

**Files:**
- Stage only the files listed in this plan.

- [x] **Step 1: Run focused regression**

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway_scheduler.py \
  tests/test_asr_gateway_metrics.py \
  tests/test_asr_gateway.py \
  tests/test_asr_deployment_scripts.py \
  tests/test_verify_asr_release.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [x] **Step 2: Run the full explicit-mock suite**

```bash
MODEL_BACKEND=mock ASR_BACKEND=mock ASR_STREAM_MODE=chunked \
ASR_REQUIRE_MODEL_MANIFEST=false ASR_MODEL_MANIFEST_PATH= \
ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8 ASR_VLLM_MAX_MODEL_LEN=65536 \
TTS_BACKEND=mock API_KEY=test-only-not-production-000000000000000000000000 \
/model/.venv/bin/python -m pytest tests -q -p no:cacheprovider
```

Expected: the full suite passes with no failures.

- [ ] **Step 3: Stage intended paths and run the commit gate**

```bash
git add -- \
  app/asr_gateway.py \
  app/asr_gateway_metrics.py \
  cloud/A10.faster-whisper.env.example \
  cloud/README-A10.md \
  docs/asr-faster-whisper-batch-coalescing-plan.md \
  scripts/verify_asr_release.sh \
  tests/test_asr_deployment_scripts.py \
  tests/test_asr_gateway.py \
  tests/test_asr_gateway_metrics.py
ln -s /model/.venv .venv
scripts/verify_asr_release.sh commit
rm .venv
```

Expected: `ASR commit verification passed`.

- [ ] **Step 4: Commit and inspect**

```bash
git commit -m "fix(asr): coalesce faster-whisper streaming batches"
git status --short
git log -2 --oneline
```

Expected: clean status and an implementation commit above design commit
`7ec11e7`.

## Task 6: Deploy the test candidate and run live acceptance

**Files:**
- No repository edits on the test host.

- [ ] **Step 1: Preserve rollback and install the candidate configuration**

On `/opt/model-test`, retain the running Qwen rollback image/environment and
update only the two selected non-secret values:

```bash
cd /opt/model-test
git pull --ff-only
cp --preserve=mode .env .env.pre-coalescing-rollback
CID="$(docker compose ps -q qwen-asr-api)"
PREVIOUS_IMAGE_ID="$(docker inspect "$CID" --format '{{.Image}}')"
docker tag "$PREVIOUS_IMAGE_ID" qwen-asr-api:pre-coalescing-rollback
sed -i -E \
  's/^ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=.*/ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=200/; s/^ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=.*/ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=6.0/' \
  .env.faster-whisper-candidate
grep -E '^ASR_GATEWAY_(SCHEDULE_MAX_WAIT_MS|MAX_SESSION_BUFFER_SECONDS)=' \
  .env.faster-whisper-candidate
```

Expected values: `200` and `6.0`. The API key and approved model paths are not
printed or rewritten.

- [ ] **Step 2: Build, verify packages, and cut over**

Build and verify the separately tagged candidate:

```bash
docker compose build qwen-asr-api
docker tag qwen-asr-api:latest qwen-asr-api:faster-whisper-coalescing-candidate
docker run --rm --entrypoint python \
  qwen-asr-api:faster-whisper-coalescing-candidate \
  -c 'import importlib.metadata as m; assert m.version("faster-whisper") == "1.2.1"; assert m.version("ctranslate2") == "4.8.1"'
docker compose stop qwen-asr-api
install -m 600 .env.faster-whisper-candidate .env
docker tag qwen-asr-api:faster-whisper-coalescing-candidate qwen-asr-api:latest
docker compose up -d --force-recreate --no-deps --no-build qwen-asr-api
for attempt in $(seq 1 120); do
  curl -fsS http://127.0.0.1:8002/ready >/dev/null && break
  sleep 5
done
curl -fsS http://127.0.0.1:8002/ready
docker compose exec -T qwen-asr-api printenv \
  ASR_BACKEND ASR_STREAM_MODE ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS \
  ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS
```

Expected: ready, `faster_whisper`, `rolling`, `200`, and `6.0`.

- [ ] **Step 3: Run two-stream acceptance before the full sweep**

From the external validation workspace, use `123.wav`, real-time pacing, 200 ms
transport chunks, HY-MT resident, and environment-only credentials:

```bash
PATH="/tmp/asr-validation-tools:$PATH" \
PYTHON_BIN="$PWD/.venv/bin/python" \
CONCURRENCY=2 AUDIO_FILE="$PWD/123.wav" \
WS_URL="${WS_URL:?set the test WebSocket URL in the environment}" \
LANGUAGE=zh CHUNK_MS=200 REALTIME=1 VERIFY_PROTOCOL=0 \
OUTPUT_DIR=/tmp/asr-fw-coalescing-c2 \
scripts/test_asr_concurrency.sh
```

Require both streams to complete, zero new failures/conflicts, a batch-fill
increase above the singleton 0.25 baseline, and current session buffer gauges
returning to zero.

- [ ] **Step 4: Run the capacity sweep**

Run the same command sequentially with `CONCURRENCY=4`, `8`, `12`, `14`, and
`16`, using a unique `/tmp` output directory per stage. Record per stage: passed/failed streams,
explicit errors, elapsed time, aggregate RTF, batch fill, batch wait, inference
latency, buffer high-water, GPU utilization, peak VRAM, and post-stage cleanup.
Stop at the first deterministic failure signature and report that boundary;
do not raise a threshold after observing a failure.

- [ ] **Step 5: Decide promotion or rollback**

Keep the candidate only if the approved evaluation threshold passes. Otherwise
restore the exact pre-change deployment:

```bash
cd /opt/model-test
docker compose stop qwen-asr-api
install -m 600 .env.pre-coalescing-rollback .env
docker tag qwen-asr-api:pre-coalescing-rollback qwen-asr-api:latest
docker compose up -d --force-recreate --no-deps --no-build qwen-asr-api
curl --retry 60 --retry-delay 5 --retry-all-errors -fsS \
  http://127.0.0.1:8002/ready
```

After readiness, require the restored backend identity and strict real-speech
verification before reopening admission.
