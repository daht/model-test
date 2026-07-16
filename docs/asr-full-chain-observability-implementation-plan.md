# ASR Full-Chain Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior. Execute this plan inline because native subagent dispatch is unavailable in the current session.

**Goal:** Implement repository-native structured ASR events, bounded aggregate metrics, precise capacity rejection reasons, timestamped evidence collection, retention, and automatic bottleneck analysis.

**Architecture:** A standard-library observability module owns safe JSON events and correlation identities. Gateway, scheduler, and faster-whisper emit correlated lifecycle and diagnostic events while bounded metric windows expose aggregate state. The monitor collects ASR/HY-MT resource evidence into one owned run directory and invokes a standard-library analyzer at finalization.

**Tech Stack:** Python 3.12, asyncio, logging, FastAPI JSON metrics, Bash, Docker Compose, nvidia-smi, pytest.

---

### Task 1: Safe structured event foundation

**Files:**
- Create: `app/asr_observability.py`
- Modify: `app/config.py`
- Create: `tests/test_asr_observability.py`

- [ ] Write tests for schema fields, diagnostic gating, deterministic batch IDs, JSON output, and sensitive-field rejection.
- [ ] Run the tests and verify they fail because the module is absent.
- [ ] Implement the minimal standard-library event emitter, `CapacityBufferError`, batch identity helper, and bounded numeric summary helper.
- [ ] Add `ASR_DIAGNOSTIC_LOGGING` and the slow-call threshold to validated settings.
- [ ] Run the focused tests and verify they pass.

### Task 2: Gateway and scheduler correlation

**Files:**
- Modify: `app/asr_gateway.py`
- Modify: `app/asr_gateway_scheduler.py`
- Modify: `app/asr_gateway_chunking.py`
- Modify: `tests/test_asr_gateway.py`
- Modify: `tests/test_asr_gateway_scheduler.py`
- Modify: `tests/test_asr_gateway_chunking.py`

- [ ] Add failing tests for exact buffer reasons and scheduler batch correlation.
- [ ] Emit always-on session terminal/release/rejection events.
- [ ] Emit diagnostic ingest, enqueue, dispatch, cleanup, and result events.
- [ ] Replace generic capacity BufferErrors with compatible controlled reasons.
- [ ] Run the focused Gateway tests.

### Task 3: Real engine group visibility and metrics

**Files:**
- Modify: `app/asr_faster_whisper.py`
- Modify: `app/asr_gateway_metrics.py`
- Modify: `app/asr_gateway.py`
- Modify: `tests/test_asr_faster_whisper.py`
- Modify: `tests/test_asr_gateway_metrics.py`

- [ ] Add failing tests for adapter group identities, actual group size, accumulated audio, elapsed time, output-size summaries, and slow-call events.
- [ ] Add bounded scheduler and engine windows with deterministic p50/p95/p99/max summaries.
- [ ] Connect the adapter engine observer to Gateway metrics without changing the worker protocol.
- [ ] Emit correlated engine start/completion events around every real engine call.
- [ ] Run faster-whisper and metrics focused tests.

### Task 4: Timestamped collector, HY-MT resource correlation, and retention

**Files:**
- Modify: `scripts/monitor_asr_bottleneck.sh`
- Modify: `tests/test_asr_deployment_scripts.py`

- [ ] Add failing script-contract tests for unique run directories, locking, atomic archives, retention, ASR/HY-MT Docker stats, GPU process samples, and analyzer invocation.
- [ ] Extend the monitor while preserving environment-only credentials and owned-path deletion rules.
- [ ] Run deployment script tests.

### Task 5: Automatic analyzer

**Files:**
- Create: `scripts/analyze_asr_bottleneck.py`
- Create: `tests/test_analyze_asr_bottleneck.py`

- [ ] Add fixed evidence fixtures for engine tails, fragmentation, capacity rejection, cleanup leaks, resource correlation, and missing samples.
- [ ] Verify analyzer tests fail before implementation.
- [ ] Implement standard-library parsing, validation, correlation, and `report.json`/`report.md` generation.
- [ ] Run analyzer tests.

### Task 6: Documentation and verification

**Files:**
- Modify: `cloud/README-A10.md`
- Modify: `docs/asr-release-verification.md`

- [ ] Document diagnostic enablement, monitor start/stop, retention controls, archive handoff, and report interpretation.
- [ ] Run all focused observability, Gateway, scheduler, adapter, metrics, analyzer, and deployment tests.
- [ ] Stage only intended files and run `scripts/verify_asr_release.sh commit`.
- [ ] Review the final diff for secrets, audio, generated evidence, and unrelated changes.
- [ ] Commit the verified candidate and provide exact test-environment rollout commands.
