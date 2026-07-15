# Semantic ASR Gateway Platform Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: use superpowers:test-driven-development for every behavior. The sole Development Agent owns all implementation tasks, review corrections, staging, verification, and commits.

**Goal:** Replace the transparent ASR proxy with a semantic, capability-driven Gateway that owns WebSocket sessions, PCM chunking, deadline scheduling, backend registration, dynamic-batch formation, serial fallback, normalized transcript events, and worker lifecycle.

**Architecture:** FastAPI remains the public I/O surface. Focused gateway modules own protocol, sessions, chunk cursors, scheduling, backend capabilities, registry, and metrics. Model/CUDA work stays behind asynchronous adapters; the initial real adapter wraps the existing serial ASR coordinator while deterministic fake adapters prove dynamic batching without GPU dependencies.

**Tech Stack:** Python 3.12, FastAPI, asyncio, Pydantic, existing ASR transcript/VAD/coordinator code, pytest 9, deterministic asyncio barriers.

---

## Execution contract

Objective: implement the design in docs/asr-gateway-platform-design.md on branch feature/asr-gateway-platform.

Non-goals:

- no backward-compatible transparent proxy mode;
- no claim that Qwen stateful streaming supports cross-session batch;
- no real A10 capacity or accuracy claim;
- no credentials, audio, model weights, generated evidence, HTML, or superpowers paths;
- no multiple CUDA owners on one GPU.

Primary risk: PCM ownership or asynchronous cleanup errors can lose, duplicate,
or publish stale session data. Every task must preserve project invariants 1-12,
especially sample conservation, transcript terminal behavior, bounded lag,
single GPU owner, and cleanup-before-publish.

Required gates:

- observe focused RED failure before production code for each task;
- focused tests after each task;
- explicit staging of intended files;
- scripts/verify_asr_release.sh commit while staged;
- commit the final candidate;
- independent read-only acceptance against the exact SHA;
- fresh primary product regression after acceptance.

Thirty-minute checkpoint: return evidence, current failing test, and supported
hypothesis if no coherent gateway core exists. Sixty-minute checkpoint: return
a testable committed subset or request scope correction.

## File map

Create:

- app/asr_gateway_backends.py: capabilities, jobs, results, adapter protocol,
  registry, leases, drain state, and local serial coordinator adapter.
- app/asr_gateway_sessions.py: session state, generations, PCM ownership cursors,
  in-flight reservation, terminal lifecycle, and session manager.
- app/asr_gateway_chunking.py: PCM ring buffer, fixed/rolling chunk policy,
  VAD ownership checks, exact boundary splitting, and ready chunk descriptions.
- app/asr_gateway_scheduler.py: per-worker ready queues, dispatcher, compatible
  batch grouping, deadlines, fairness, reservations, and result application.
- app/asr_gateway_protocol.py: public commands, normalized events, validation,
  and backend-independent error codes.
- app/asr_gateway_metrics.py: stage timestamps, counters, worker and scheduler
  snapshots without raw audio or credentials.

Replace:

- app/asr_gateway.py: semantic FastAPI lifespan, WebSocket handler, health,
  readiness, backend inventory, and integration wiring.
- tests/test_asr_gateway.py: public semantic gateway tests instead of transparent
  forwarding tests.

Add focused tests:

- tests/test_asr_gateway_backends.py
- tests/test_asr_gateway_sessions.py
- tests/test_asr_gateway_chunking.py
- tests/test_asr_gateway_scheduler.py
- tests/test_asr_gateway_protocol.py

Modify:

- app/config.py: validated Gateway scheduler, queue, update, and drain settings.
- docker-compose.yml: make the ASR service start the semantic Gateway app while
  retaining one process, one Uvicorn worker, and one model owner on the A10.
- Delete requirements-asr-gateway.txt and Dockerfile.asr-gateway because V1
  deliberately runs the selected local adapter in the ASR runtime process.
- Delete docker-compose.asr-multipod.yml because the old two-model experiment is
  not part of the new Gateway architecture.
- README.md, cloud/README-A10.md, docs/asr-multipod-gateway.md: describe the new
  semantic Gateway and explicitly mark A10 capacity as unverified.

### Task 1: Backend capability and registry contract

Files:

- Create app/asr_gateway_backends.py
- Create tests/test_asr_gateway_backends.py

- [ ] Step 1: write failing tests for validated capabilities.

Create tests that instantiate BackendCapabilities with protocol version, worker
ID, immutable model ID/revision, streaming mode, dispatch mode, VAD mode,
preferred chunk duration, batch item/audio limits, in-flight limits, session
capacity, retry safety, and result mode. Assert rejection of empty IDs,
non-positive limits, single mode with max_batch_items above one, simultaneous
Gateway/Worker VAD ownership, and changing immutable identity on re-registration.

- [ ] Step 2: run the focused tests and verify RED.

Run:

    .venv/bin/python -m pytest tests/test_asr_gateway_backends.py -q

Expected: collection fails because app.asr_gateway_backends does not exist.

- [ ] Step 3: implement the minimal capability types and registry.

Define string enums for StreamingMode, DispatchMode, VadMode, ResultMode, and
WorkerLifecycle. Define frozen BackendCapabilities and validated BackendSnapshot.
Define BackendRegistry.register, mark_ready, acquire, release, begin_drain,
remove, and snapshot. Registry acquire must exclude unready/draining workers and
return a lease whose release is idempotent.

Use explicit ValueError messages that name the invalid field. Never store raw
audio or credentials in registry snapshots.

- [ ] Step 4: add RED tests for drain-and-switch and lease accounting.

Use two workers with the same model revision. Prove new leases select the ready
replacement after begin_drain, existing leases remain attributed to the old
worker, duplicate release cannot underflow, and removal with active leases fails.

- [ ] Step 5: implement drain and lease behavior, then run GREEN.

Run the focused test command until all tests pass.

- [ ] Step 6: commit the contract.

Stage only the two files and commit with:

    feat(asr): add gateway backend registry contract

### Task 2: Session ownership and PCM chunk management

Files:

- Create app/asr_gateway_sessions.py
- Create app/asr_gateway_chunking.py
- Create tests/test_asr_gateway_sessions.py
- Create tests/test_asr_gateway_chunking.py

- [ ] Step 1: write failing PCM conservation tests.

Test aligned pcm_s16le append, accepted/scheduled/acknowledged cursors, exact
split at a maximum sample boundary, retained remainder, finish flush, and
reservation rollback. Assert every accepted sample is exactly one of buffered,
reserved, acknowledged, or explicitly discarded.

- [ ] Step 2: verify RED using:

    .venv/bin/python -m pytest tests/test_asr_gateway_sessions.py tests/test_asr_gateway_chunking.py -q

Expected: missing module failures.

- [ ] Step 3: implement PcmRingBuffer, ChunkPolicy, ReadyChunk, GatewaySession,
and SessionManager.

PcmRingBuffer stores aligned bytes with absolute sample cursors and supports
append, reserve_range, acknowledge, rollback, and bounded compaction.
ChunkPolicy derives minimum/preferred/maximum samples from worker capabilities.
GatewaySession owns generation, selected worker, backend session, deadlines,
terminal state, transcript state, one in-flight reservation, and optional
Gateway VAD detector.

Reject odd PCM byte lengths, cursor regressions, double reservations, buffer
overflow, and conflicting VAD ownership.

- [ ] Step 4: write failing lifecycle and stale-generation tests.

Prove one in-flight job per session, new audio coalesces while busy, closing
increments or invalidates generation, an old result cannot acknowledge samples,
finish schedules remaining audio once, and abort releases reservations once.

- [ ] Step 5: implement lifecycle behavior and run GREEN.

Run both focused files. Then run:

    .venv/bin/python -m pytest tests/test_asr_vad.py tests/test_asr_streaming.py -q

Expected: all pass with existing sample and transcript invariants intact.

- [ ] Step 6: commit with:

    feat(asr): add gateway session and chunk ownership

### Task 3: Deadline-aware dispatcher and dynamic batch scheduler

Files:

- Create app/asr_gateway_scheduler.py
- Create tests/test_asr_gateway_scheduler.py
- Modify app/asr_gateway_backends.py to add the adapter result channel and
  worker-acceptance lifecycle used by scheduler cleanup

- [ ] Step 1: write a deterministic RED test for real batch formation.

Use a FakeClock, asyncio.Event barriers, three sessions, one ready dynamic worker,
identical batch keys, max_batch_items four, and max wait twenty milliseconds.
Release all sessions through a barrier and assert the adapter receives one list
of three jobs rather than three calls.

- [ ] Step 2: verify RED.

Run:

    .venv/bin/python -m pytest tests/test_asr_gateway_scheduler.py::test_compatible_ready_sessions_form_one_dynamic_batch -q

Expected: missing scheduler module.

- [ ] Step 3: implement InferenceJob, BatchKey, InferenceResult,
WorkerAdapter protocol, GatewayScheduler, and one scheduling iteration.

Scheduling must select a worker before batching, group only identical batch
keys, apply item and total-audio cost limits, select earliest deadlines, permit
one job per session per round, dispatch immediately when full, and expose an
event-driven wake method. Tests call a deterministic run_once method; production
uses an asyncio task and deadline timer.

- [ ] Step 4: write RED tests for serial fallback, timeout, fairness, and
incompatible keys.

Prove:

- single mode always submits length-one lists;
- a partial dynamic batch dispatches at the fake deadline without sleep;
- one session cannot appear twice in one round;
- different language/decoding/length-bucket keys do not merge;
- full batch dispatches before max wait;
- queue and audio reservations reject before unbounded growth.

- [ ] Step 5: implement the minimal behavior and run GREEN.

Run the complete scheduler test file.

- [ ] Step 6: write RED tests for result isolation and cleanup ordering.

Test success, per-item failure, whole-batch failure, cancellation before worker
acceptance, cancellation after acceptance, stale generation, worker loss, and
reservation release exactly once. Successful results must become publishable
only after cleanup accounting completes.

- [ ] Step 7: implement result handling and run GREEN.

Also run session/chunk/backend tests together.

- [ ] Step 8: commit with:

    feat(asr): add deadline dynamic batch scheduler

### Task 4: Public protocol and normalized transcript events

Files:

- Create app/asr_gateway_protocol.py
- Create tests/test_asr_gateway_protocol.py
- Reuse app/asr_streaming.py without duplicating transcript concatenation

- [ ] Step 1: write failing protocol tests.

Cover start command, language/options normalization, pcm_s16le/16k/mono
validation, explicit segment, finish, abort, sequence continuity, partial
replacement, sentence_final append, one final, no event after final, confirmed
prefix conflict, and error payload sanitization.

- [ ] Step 2: verify RED.

Run the protocol test file and observe the missing module failure.

- [ ] Step 3: implement public command models, event serialization, and result
normalization around StreamingTranscriptState.

Adapter results support cumulative_snapshot, replaceable_segment, and
confirmed_plus_tail. Do not maintain a second independent transcript
concatenation implementation.

- [ ] Step 4: run GREEN and existing transcript/client regression.

Run:

    .venv/bin/python -m pytest tests/test_asr_gateway_protocol.py tests/test_asr_streaming.py tests/test_stream_asr_client.py -q

Expected: all pass.

- [ ] Step 5: commit with:

    feat(asr): add gateway protocol normalization

### Task 5: Metrics and readiness snapshots

Files:

- Create app/asr_gateway_metrics.py
- Create tests/test_asr_gateway_metrics.py
- Modify app/asr_gateway_scheduler.py
- Modify app/asr_gateway_backends.py

- [ ] Step 1: write failing metric accounting tests.

Create a job timeline with all nine required stages. Assert derived chunk wait,
batch wait, worker wait, inference time, egress time, decoded seconds, aggregate
RTF, batch fill ratio, queue depth, and active session counts.

Assert snapshots contain no PCM bytes, authorization fields, API keys, or
transcript text.

- [ ] Step 2: verify RED, implement immutable metric records and bounded
aggregates, then run GREEN.

Use monotonic numeric timestamps passed by callers. Metrics code must not call
wall-clock time in deterministic tests.

- [ ] Step 3: add readiness tests.

Readiness is false with no worker, un-warmed worker, draining-only worker, or no
capacity. It is true with at least one warmed accepting worker satisfying the
default route.

- [ ] Step 4: implement readiness aggregation and commit with:

    feat(asr): add gateway scheduling metrics

### Task 6: Local serial adapter for the current Qwen coordinator

Files:

- Modify app/asr_gateway_backends.py or create app/asr_gateway_local_adapter.py
- Create tests/test_asr_gateway_local_adapter.py
- Reuse app/asr_inference.py and app/asr.py without changing the pinned runtime
  contract

- [ ] Step 1: write failing adapter tests using a fake ASRInferenceCoordinator.

Prove warmup/start, open session, single-item submit calling add_audio, segment
finish, input finish, abort, result conversion, and snapshot capacity. Assert
length above one is rejected and capabilities advertise dispatch_mode single,
max_batch_items one, stateful stickiness, and the selected VAD/result mode.

- [ ] Step 2: verify RED.

Run the focused adapter test file.

- [ ] Step 3: implement LocalCoordinatorAdapter.

The adapter must not import or load qwen_asr during module import. It receives a
coordinator factory, starts and stops it in lifecycle methods, maps backend
session IDs, and sanitizes errors. CUDA/model work remains in the coordinator
owner thread.

- [ ] Step 4: write and pass tests for coordinator failure, timeout, stale
session, abort cleanup, and readiness loss.

- [ ] Step 5: run existing inference and Qwen contract regression.

Run:

    .venv/bin/python -m pytest tests/test_asr_gateway_local_adapter.py tests/test_asr_inference.py tests/test_asr_api.py -q

Expected: all pass.

- [ ] Step 6: commit with:

    feat(asr): adapt serial coordinator to gateway workers

### Task 7: Semantic FastAPI Gateway integration

Files:

- Replace app/asr_gateway.py
- Replace tests/test_asr_gateway.py
- Modify app/config.py
- Modify docker-compose.yml
- Delete requirements-asr-gateway.txt
- Delete Dockerfile.asr-gateway

- [ ] Step 1: write failing public integration tests.

Using fake serial and dynamic adapters, test authentication before admission,
start timeout, ready event, binary PCM ingestion without awaiting inference,
multiple connections forming a batch, result routing to the correct session,
explicit segment, finish, close 1000, disconnect abort, overload 1013, backend
loss, and exactly one terminal event.

Use deterministic barriers for simultaneous clients. Do not use sleeps.

- [ ] Step 2: verify RED against the old transparent gateway.

Expected failures must show the old proxy does not own semantic sessions or
batch scheduling.

- [ ] Step 3: replace app/asr_gateway.py with semantic wiring.

Lifespan creates registry, session manager, chunk manager, metrics, local adapter,
and scheduler. WebSocket handler authenticates, parses start, allocates a sticky
worker, appends PCM, signals ready chunks, consumes result events, and finalizes
with explicit cleanup.

Expose:

    GET /health
    GET /ready
    GET /v1/transcribe/stream-info
    GET /v1/asr/backends
    GET /v1/asr/metrics
    WS  /v1/transcribe/stream

Backend inventory and metrics require authentication if they expose operational
details. No raw audio or credentials are returned.

- [ ] Step 4: add validated Settings fields.

Add bounded settings for schedule max wait, ready jobs, queued audio, session
buffer, update interval, drain timeout, and default backend route. Enforce
positive finite values and ensure buffered audio cannot exceed lag limits.

- [ ] Step 5: update runtime packaging.

Delete the transparent-proxy-only Gateway requirements and Dockerfile. Change
the ASR service command in docker-compose.yml to run app.asr_gateway:app with
exactly one Uvicorn worker. The semantic Gateway uses the existing ASR image and
LocalCoordinatorAdapter, so the same process owns one selected model and one
GPU owner thread. Remove the upstream websockets dependency when no remaining
tracked code imports it. Do not leave a CPU-only Gateway claim.

- [ ] Step 6: run focused integration GREEN.

Run:

    .venv/bin/python -m pytest tests/test_asr_gateway.py tests/test_asr_gateway_backends.py tests/test_asr_gateway_sessions.py tests/test_asr_gateway_chunking.py tests/test_asr_gateway_scheduler.py tests/test_asr_gateway_protocol.py tests/test_asr_gateway_metrics.py tests/test_asr_gateway_local_adapter.py -q

- [ ] Step 7: commit with:

    feat(asr): replace proxy with semantic gateway

### Task 8: Lifecycle, documentation, and deterministic release regression

Files:

- Modify README.md
- Modify cloud/README-A10.md
- Replace or remove docs/asr-multipod-gateway.md
- Delete docker-compose.asr-multipod.yml
- Modify docker-compose.yml
- Modify tests/test_asr_deployment_scripts.py
- Modify tests/test_asr_gateway.py
- Modify tests/test_verify_asr_release.py only if the authoritative gate surface
  genuinely changes

- [ ] Step 1: write failing documentation/deployment contract tests.

Assert the old two-model-on-one-A10 experiment and its Compose file are removed.
Assert docker-compose.yml starts the semantic Gateway with one worker and one
model owner per GPU, new configuration, readiness/warmup, drain-and-switch
model lifecycle, and explicit absence of production/capacity claims.

- [ ] Step 2: verify RED and update deployment/docs minimally.

Document a test deployment with one A10, one selected model worker, one Gateway
entrypoint, and environment-only credentials. Describe backend plugin testing
and external A10 gates.

- [ ] Step 3: run focused deployment and Gateway tests.

Run:

    .venv/bin/python -m pytest tests/test_asr_deployment_scripts.py tests/test_asr_gateway.py tests/test_verify_asr_release.py -q

- [ ] Step 4: run the complete explicit-mock suite.

Run the same C01 command through the repository commit gate rather than placing
an API key in the command.

- [ ] Step 5: self-review the entire delta.

Check design coverage, no placeholders, no duplicated transcript state machine,
no raw audio logs, no secret-bearing arguments, bounded queues, explicit worker
loss, one in-flight job per session, and no compatibility code for the old
transparent proxy.

- [ ] Step 6: stage only intended files and run the authoritative commit gate.

Run:

    git add -- <explicit intended paths>
    scripts/verify_asr_release.sh commit

Expected: every commit gate passes while the candidate is staged.

- [ ] Step 7: commit the final candidate.

Commit message:

    feat(asr): implement semantic gateway platform

Return DONE only with focused test counts, complete test count, commit-gate
output summary, final SHA, clean candidate status, and external A10/release/live
gaps.
