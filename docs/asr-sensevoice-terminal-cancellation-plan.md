# SenseVoice Terminal Generation Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. ASR delivery still requires one Development Agent to own every implementation and review correction.

**Goal:** Prevent queued jobs from a session generation from reaching the adapter after an internal result path terminates that session.

**Architecture:** Reuse the scheduler's existing generation cancellation set and cancelled-queue rejection path. Register cancellation in the common `_fail_session()` transition before `GatewaySession.fail()` invalidates the generation; do not wait on the accepted barrier from result publication.

**Tech Stack:** Python 3.12, asyncio, pytest, existing GatewayRuntime and GatewayScheduler.

---

### Task 1: Prove the post-terminal dispatch race

**Files:**
- Modify: `tests/test_asr_gateway.py`

- [ ] **Step 1: Add a deterministic failing regression**

Add a test that uses `FakeClock`, `FakeAdapter`, and a wrapped scheduler cleanup callback. The wrapper acknowledges the first job and then enqueues a continuation before the first result is published:

```python
def test_connection_lag_cancels_continuation_before_adapter_submit(monkeypatch):
    class Emitter:
        def __init__(self):
            self.records = []

        def emit(self, event, *, component, **fields):
            self.records.append({"event": event, "component": component, **fields})

    async def scenario():
        clock = FakeClock()
        adapter = FakeAdapter()
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=1_000,
            asr_max_connection_lag_seconds=1,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter}, clock=clock)
        emitter = Emitter()
        monkeypatch.setattr("app.asr_gateway.observability_events", lambda: emitter)
        await adapter.warmup()
        await runtime.registry.register(adapter.capabilities)
        await runtime.registry.mark_ready("fake", True)
        session = await runtime.open_session("s", language="zh", options={})
        context = runtime._contexts[session.session_id]
        await runtime.ingest(session, b"\x01\x00" * 4, force=True)
        original_cleanup = runtime.scheduler.cleanup

        async def cleanup_then_enqueue(job):
            await original_cleanup(job)
            await runtime.ingest(session, b"\x02\x00" * 4, force=True)

        runtime.scheduler.cleanup = cleanup_then_enqueue
        clock.advance(2)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.scheduler.run_once("fake", force=True)
        snapshot = runtime.scheduler.snapshot()
        accounting = context.session.sample_accounting
        terminal = [
            item for item in emitter.records
            if item["event"] == "asr_session_terminal"
        ]
        conflicts = [
            item for item in emitter.records
            if item["event"] == "asr_cleanup_conflict"
        ]
        return adapter, snapshot, accounting, terminal, conflicts

    adapter, snapshot, accounting, terminal, conflicts = asyncio.run(scenario())

    assert len(adapter.calls) == 1
    assert snapshot["ready_depth"] == snapshot["queued_samples"] == 0
    assert accounting["buffered"] == accounting["reserved"] == 0
    assert [item["reason"] for item in terminal] == ["connection_lag"]
    assert conflicts == []
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py::test_connection_lag_cancels_continuation_before_adapter_submit \
  -q -p no:cacheprovider
```

Expected: FAIL because `adapter.calls` contains both the original job and the stale continuation.

### Task 2: Cancel the generation at the common failure transition

**Files:**
- Modify: `app/asr_gateway.py`
- Test: `tests/test_asr_gateway.py`

- [ ] **Step 1: Add the minimal implementation**

Capture the generation and synchronously register scheduler cancellation before invalidating the session:

```python
        session_id = ctx.session.session_id
        generation = ctx.session.generation
        self.scheduler.cancel_session(session_id, generation=generation)
        ctx.session.fail()
```

Use `session_id` in the following adapter abort and terminal event. Do not call
`wait_session_safe()` in `_fail_session()`.

- [ ] **Step 2: Verify GREEN and focused invariants**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py::test_connection_lag_cancels_continuation_before_adapter_submit \
  tests/test_asr_gateway.py::test_cancelled_queued_timeline_is_settled_once \
  tests/test_asr_gateway.py::test_cancelled_dispatched_timeline_is_settled_once \
  -q -p no:cacheprovider
```

Expected: `3 passed`; no hang and no cleanup conflict.

- [ ] **Step 3: Run the complete focused files**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py tests/test_asr_gateway_scheduler.py \
  -q -p no:cacheprovider
```

Expected: all tests pass.

### Task 3: Stage, gate, and commit the candidate

**Files:**
- Modify: `app/asr_gateway.py`
- Modify: `tests/test_asr_gateway.py`

- [ ] **Step 1: Check the exact delta**

Run `git diff --check`, inspect `git diff -- app/asr_gateway.py tests/test_asr_gateway.py`, and confirm no unrelated file changed.

- [ ] **Step 2: Stage only intended implementation files**

```bash
git add -- app/asr_gateway.py tests/test_asr_gateway.py
```

- [ ] **Step 3: Run the staged commit gate**

Ensure the ignored worktree `.venv` points to the repository test environment, then run:

```bash
scripts/verify_asr_release.sh commit
```

Expected: every commit gate passes, including the complete explicit-mock pytest suite.

- [ ] **Step 4: Commit and report evidence**

```bash
git commit -m "fix(asr): cancel queued work before session failure"
git status --short
git rev-parse HEAD
```

Expected: the candidate worktree is clean and the returned SHA identifies the exact independently testable candidate.
