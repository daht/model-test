# SenseVoice Batch Failure Containment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` and follow the repository ASR
> production-delivery workflow. One Development Agent owns all implementation
> and review corrections.

**Goal:** Classify the first SenseVoice batch failure safely and contain a fatal
worker submission failure without further dispatch, duplicate terminal events,
or cleanup/result conflicts.

**Architecture:** SenseVoice supplies stable, non-secret failure stages at the
engine boundary. The scheduler owns fail-stop and queued-work settlement because
it owns accepted and queued jobs. The gateway performs an idempotent session
terminal transition only after scheduler ownership cleanup.

**Tech Stack:** Python 3.12, asyncio, pytest, FastAPI gateway scheduler,
SenseVoice/FunASR adapter, structured ASR observability events.

---

### Task 1: Safe SenseVoice first-failure classification

**Files:**

- Modify: `app/asr_sensevoice.py`
- Test: `tests/test_asr_sensevoice.py`

- [ ] **Step 1: Write failing engine-contract tests**

Add tests that make the fake FunASR model raise a private exception, return the
wrong result count, and return a malformed item. Assert stable stages and safe
exception types without exposing private messages or fake raw output:

```python
def test_engine_classifies_generate_failure_without_private_text(...):
    with pytest.raises(SenseVoiceBatchFailure) as failure:
        engine.transcribe_batch([np.ones(16, dtype=np.float32)], language="zh")
    assert failure.value.stage == "engine_generate"
    assert failure.value.exception_type == "PrivateEngineError"
    assert "private-model-details" not in str(failure.value)


def test_engine_classifies_result_count_failure(...):
    with pytest.raises(SenseVoiceBatchFailure) as failure:
        engine.transcribe_batch(
            [np.ones(16, dtype=np.float32), np.ones(16, dtype=np.float32)],
            language="zh",
        )
    assert failure.value.stage == "result_count"


def test_engine_classifies_result_contract_failure(...):
    with pytest.raises(SenseVoiceBatchFailure) as failure:
        engine.transcribe_batch([np.ones(16, dtype=np.float32)], language="zh")
    assert failure.value.stage == "result_contract"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_asr_sensevoice.py::test_engine_classifies_generate_failure_without_private_text \
  tests/test_asr_sensevoice.py::test_engine_classifies_result_count_failure \
  tests/test_asr_sensevoice.py::test_engine_classifies_result_contract_failure
```

Expected: FAIL because `SenseVoiceBatchFailure` and stable stages do not exist.

- [ ] **Step 3: Implement the minimal classified exception**

Add one internal exception carrying only stable safe fields:

```python
class SenseVoiceBatchFailure(RuntimeError):
    def __init__(self, stage: str, exception_type: str) -> None:
        super().__init__(f"SenseVoice batch failed at {stage}")
        self.stage = stage
        self.exception_type = exception_type
```

Wrap only the FunASR call and existing result validation. Convert internal model
exceptions to `engine_generate`, non-list/malformed items to `result_contract`,
and length mismatch to `result_count`. Preserve the original exception as the
Python cause without logging it.

- [ ] **Step 4: Write and verify the adapter event RED test**

Use the repository event sink with a failing engine and assert exactly one
`asr_engine_group_failed` event with:

```python
assert event["failure_stage"] == "engine_generate"
assert event["exception_type"] == "PrivateEngineError"
assert event["group_size"] == 2
assert event["final_items"] == 0
assert event["accumulated_audio_seconds"] == pytest.approx(0.002)
assert event["min_input_audio_seconds"] == pytest.approx(0.001)
assert event["max_input_audio_seconds"] == pytest.approx(0.001)
assert "private-model-details" not in json.dumps(event)
```

Run the test before adding the event and observe failure because the event is
absent. Then emit the event in `SenseVoiceAdapter.submit()` immediately around
the engine group call and re-raise the classified failure.

- [ ] **Step 5: Run the SenseVoice focused suite**

Run:

```bash
.venv/bin/pytest -q tests/test_asr_sensevoice.py
```

Expected: all tests pass with no raw private failure text in captured output.

### Task 2: Scheduler fail-stop and queued-work settlement

**Files:**

- Modify: `app/asr_gateway_scheduler.py`
- Test: `tests/test_asr_gateway_scheduler.py`

- [ ] **Step 1: Write the deterministic fatal-worker RED test**

Create a controlled adapter whose first `submit()` raises and enqueue two
batches before calling `run_once()`. Record submit calls, cleanup/reject calls,
published results, and worker-failure callbacks. Assert:

```python
assert adapter.calls == [["first-1"]]
assert failures == [("worker-1", "submit_failed")]
assert sorted(cleaned + rejected) == ["first-1", "queued-1"]
assert {item.job_id for item in published} == {"first-1", "queued-1"}
assert all(item.error == "RuntimeError: batch failed" for item in published)
assert snapshot["queued_samples"] == 0
assert snapshot["ready_depth"] == 0
```

Run:

```bash
.venv/bin/pytest -q \
  tests/test_asr_gateway_scheduler.py::test_submit_failure_halts_worker_and_settles_queued_jobs
```

Expected: FAIL because current code dispatches the queued job on the next
iteration.

- [ ] **Step 2: Implement minimal worker fail-stop state**

Add a private failed-worker set to `GatewayScheduler`. On the first submission
exception, mark the worker failed before invoking `worker_failed`, and ensure
subsequent `run_once()` calls never invoke that adapter again. Report the worker
failure callback once.

- [ ] **Step 3: Settle queued work without adapter submission**

Remove queued jobs for the failed worker, decrement queued samples exactly once,
rollback their reservations through `reject`, and publish sanitized error
results only after rollback. Stale rollback uses the existing `discard` path.
Ensure accepted jobs complete normal `cleanup` before their fatal result is
published and their safe barriers are set.

- [ ] **Step 4: Verify scheduler failure and cancellation suites**

Run:

```bash
.venv/bin/pytest -q tests/test_asr_gateway_scheduler.py
```

Expected: all scheduler tests pass, including inference timeout and cancellation
barriers.

### Task 3: Idempotent gateway failure terminal

**Files:**

- Modify: `app/asr_gateway.py`
- Test: `tests/test_asr_gateway.py`

- [ ] **Step 1: Write the multi-session batch-failure RED test**

Use a dynamic fake adapter with one failing batch and queued sessions behind it.
Run scheduler settlement and send outbound events. Assert:

```python
assert adapter.submit_calls == 1
assert terminal_counts == {session_id: 1 for session_id in session_ids}
assert metrics["conflicts"] == 0
assert cleanup_conflicts == []
assert runtime._contexts == {}
assert adapter.sessions == set()
assert all(worker["active_leases"] == 0 for worker in backend_workers.values())
```

Run the isolated test and observe RED from duplicate terminal/conflict behavior.

- [ ] **Step 2: Make failure transition idempotent**

Change `_fail_session()` to return whether it performed the transition. If the
session is already `FAILED`, `ABORTED`, or `SUCCEEDED`, do not abort again and do
not emit another `asr_session_terminal`. Callers enqueue a terminal protocol
error only when they own the first transition.

- [ ] **Step 3: Discard stale terminal results**

Before applying a published result, treat an already terminal protocol/session
or stale generation as discard-only. Do not count a result conflict and do not
invoke `_fail_session()` again. Preserve existing confirmed-prefix conflict
behavior for live, matching-generation sessions.

- [ ] **Step 4: Run gateway and cleanup regression suites**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_asr_gateway.py \
  tests/test_asr_gateway_sessions.py \
  tests/test_asr_gateway_local_adapter.py
```

Expected: all tests pass with zero unexpected conflict events.

### Task 4: Monitoring parser and complete candidate gate

**Files:**

- Modify: `scripts/analyze_asr_bottleneck.py`
- Test: `tests/test_asr_monitoring.py`

- [ ] **Step 1: Add a monitoring regression for the new failure event**

Supply an `asr_engine_group_failed` fixture and assert the generated report
retains safe stage, exception type, batch ID, group size, and accumulated audio
seconds without requiring raw exception text.

- [ ] **Step 2: Run monitoring and affected focused suites**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_asr_monitoring.py \
  tests/test_asr_sensevoice.py \
  tests/test_asr_gateway_scheduler.py \
  tests/test_asr_gateway.py
```

Expected: all tests pass.

- [ ] **Step 3: Stage only intended files and run the commit gate**

Stage explicitly:

```bash
git add -- \
  app/asr_sensevoice.py \
  app/asr_gateway_scheduler.py \
  app/asr_gateway.py \
  scripts/analyze_asr_bottleneck.py \
  tests/test_asr_sensevoice.py \
  tests/test_asr_gateway_scheduler.py \
  tests/test_asr_gateway.py \
  tests/test_asr_monitoring.py
```

Run:

```bash
ASR_MODEL_NAME=Qwen3-ASR-1.7B \
ASR_MAX_UTTERANCE_SECONDS=30.0 \
scripts/verify_asr_release.sh commit
```

Expected: all C01-C06 commit gates pass.

- [ ] **Step 4: Commit the candidate**

Commit with:

```bash
git commit -m "fix(asr): contain SenseVoice batch failures"
```

Return the exact SHA, focused-test counts, commit-gate count, RED evidence, and
external A10/live gaps to the primary agent.

### Task 5: Independent acceptance and product regression

**Files:** None; verification is read-only.

- [ ] **Step 1: Independent Test Agent acceptance**

Against a detached worktree at the committed SHA, reproduce test sensitivity,
run affected adversarial/focused tests and `scripts/verify_asr_release.sh commit`,
scan secrets/forbidden artifacts/residue, and return `ACCEPTED` or `REJECTED`.

- [ ] **Step 2: Primary product regression**

After `ACCEPTED`, the primary agent reruns the deterministic fatal-worker probe,
the focused suites, and the commit gate on the accepted SHA. Confirm one
terminal per session, zero post-failure dispatch, zero cleanup/result conflicts,
and clean tracked Git state.

- [ ] **Step 3: Integrate and report external gap**

Fast-forward the accepted branch into `main`. Report the committed SHA and local
evidence. A10 deployment and the monitored 88-stream run remain unexecuted until
the user deploys the accepted SHA.
