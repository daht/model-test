# ASR Gateway Platform Design

Date: 2026-07-15

## Objective

Replace the experimental transparent ASR proxy with a semantic ASR Gateway
platform. The gateway owns the public WebSocket contract, session lifecycle,
audio buffering, chunk readiness, deadline-aware scheduling, backend selection,
and normalized transcript events. Model execution is isolated behind a
versioned adapter contract so Qwen3-ASR, faster-whisper, SenseVoice, Paraformer,
and future runtimes can be evaluated without changing the public protocol.

The first deployment targets one NVIDIA A10 and one selected model worker. The
interfaces support multiple registered workers and GPU-aware dispatch from the
first version.

## Non-goals

- Preserve compatibility with the old transparent proxy or multi-Pod topology.
- Implement unsupported Qwen3-ASR stateful cross-session batching.
- Claim capacity, accuracy, or production readiness without real A10 evidence.
- Run blocking preprocessing, model loading, or CUDA inference on the FastAPI
  event loop.
- Keep a stateful session alive after its assigned worker is lost.
- Load multiple large ASR models concurrently on one A10 by default.

## Success criteria

1. One public WebSocket protocol serves every backend.
2. Sessions preserve PCM ownership, bounded buffering, strict ordering, and one
   terminal final event.
3. A deadline-aware scheduler forms dynamic batches for capable workers and
   automatically uses batch size one for serial workers.
4. Workers advertise validated capabilities, readiness, load, model identity,
   and capacity through a registry.
5. Stateful sessions remain sticky and execute strictly in order.
6. Worker loss, overload, cancellation, stale results, and malformed results
   fail explicitly without corrupting another session.
7. Deterministic tests prove batching, serial fallback, fairness, ordering,
   cancellation barriers, state isolation, and protocol terminal behavior.
8. Metrics distinguish ingress, chunking, scheduling, worker queue, inference,
   and egress bottlenecks.

## Five-layer architecture

    Client WebSockets
            |
            v
    1. Gateway / public protocol
            |
            v
    2. Session Manager
            |
            v
    3. Chunk Manager
            |
            v
    4. Ready Queue + GPU Dispatcher + Batch Scheduler
            |
            v
    5. Inference Adapter / Backend Worker

The gateway is semantic, not transport-transparent. It parses public commands,
owns session state, chooses a worker, validates results, and emits normalized
events. A worker never speaks the public WebSocket protocol directly.

## 1. Gateway

The FastAPI layer owns authentication, public WebSocket lifecycle, start and
frame validation, connection limits, explicit segment, finish, abort,
disconnect, normalized events, health, readiness, backend inventory, and
metrics.

The receive loop validates and appends audio without waiting for each inference
call. A separate result path publishes events in order. All queues are bounded.
Each live session has exactly one ordered outbound queue and one sender task.
Protocol mutation is serialized by the session lock; inference, segment,
finish, final, and terminal error paths enqueue envelopes through that same
queue. Only the sender calls `send_json` or closes the established socket, so a
pending partial cannot be overtaken by control output and no event follows a
terminal final or error.

## 2. Session Manager

Each GatewaySession contains:

    gateway_session_id and generation
    selected_worker_id and backend_session_id
    language and decoding options
    PCM ring buffer
    accepted, scheduled, and acknowledged sample cursors
    next update deadline
    inference-in-flight flag
    pending control action
    transcript state
    connection and session deadlines
    terminal state

The gateway owns PCM until a worker accepts the job reservation. Audio cursors
prove samples are neither lost nor duplicated. Each session has at most one
in-flight inference job. New audio arriving during inference is coalesced into
the next job.

Every result includes session generation, job sequence, and sample range.
Stale, cancelled, closed-session, old-generation, or invalid-range results are
never published.

## 3. Chunk Manager

The common chunk manager owns PCM accounting and creates work on these triggers:

- preferred update chunk reached;
- VAD endpoint;
- explicit segment;
- exact maximum utterance boundary;
- end of input.

Policy is capability-driven. A backend declares stateful, chunked,
rolling-window, or offline mode, plus preferred chunk duration, overlap,
maximum input duration, and VAD ownership.

Only one component may discard or segment samples. Gateway VAD sends only
approved audio. Worker VAD receives all PCM and reports decoded or discarded
progress. Both sides enabling VAD is invalid.

Rolling adapters may include acknowledged context, but must report newly
consumed samples separately so repeated context is never counted twice.

## 4. GPU Dispatcher and Batch Scheduler

Worker selection happens before batching because a batch cannot cross model
instances or GPUs. The dispatcher filters on readiness, model revision,
language and feature support, streaming mode, capacity, and device constraints.

Stateful sessions are sticky for their lifetime. Stateless work may be rerouted
only before worker acceptance and only when declared retry-safe.

Each worker has a ready queue. Scheduling is event-driven: ready jobs, available
worker slots, full batches, and deadline timers wake the scheduler. The 10-20 ms
setting is maximum micro-batch collection delay, not a polling loop or model
update interval.

The scheduler uses:

- earliest deadline first;
- at most one job per session per round;
- tenant and session quotas;
- maximum batch item and total cost limits;
- compatible batch-key grouping;
- immediate dispatch when full or deadline-bound;
- batch-size-one fallback for non-batching workers.

The batch key includes worker and model revision, language mode, task,
timestamps, prompt or hotword identity, decoding parameters, sample format, and
input-length bucket. Adapters may extend it.

Dispatch modes are single, fixed_microbatch, dynamic_microbatch, continuous,
and stateful_dynamic. Continuous workers own internal admission and removal;
the gateway does not impose a request-response batch boundary on them.

## 5. Inference Adapter and Backend Worker

The asynchronous contract provides:

    capabilities
    warmup
    open_session
    submit
    finish_session
    abort_session
    cancel
    drain
    close
    snapshot

Submit always accepts a list. Serial adapters accept length one. Micro-batch
adapters execute the list as one model call. Continuous adapters admit work into
their engine and return through the same asynchronous result channel.

Workers own model objects, CUDA state, backend streaming state, and
model-specific preprocessing. Blocking work runs on a dedicated owner thread or
process. One model owner serves one GPU unless a separately tested backend
contract proves otherwise.

Capabilities include protocol version, worker and immutable model identity,
GPU identity, languages, tasks, streaming and dispatch mode, stateful and
retry-safe flags, VAD and result mode, chunk limits, overlap, batch item and
cost limits, in-flight limits, and session and queue capacity.

Initial acceptance uses deterministic fake serial and dynamic-batch adapters,
plus a serial wrapper around the pinned Qwen streaming runtime. Qwen advertises
single until a separate real-runtime batch implementation passes A10 gates.

## Registry and model lifecycle

The registry maintains worker identity, capabilities, readiness, load, and
leases. Registration requires schema validation and non-silent warmup.
Duplicate identities, changing immutable model identity, and invalid capacity
fail closed.

Reload uses drain-and-switch:

1. Start and warm the new worker.
2. Register it for new sessions.
3. Mark the old worker draining.
4. Stop routing new sessions to the old worker.
5. Let existing stateful sessions finish within a bounded deadline.
6. Explicitly abort remaining sessions after the deadline.
7. Unload and remove the old worker.

Control-plane methods for register, drain, reload, and remove are reserved and
separately authenticated. V1 may use static startup registration through the
same registry API.

## Transcript normalization

Adapters return cumulative snapshots, replaceable segment snapshots, or
confirmed segment plus replaceable tail. Gateway transcript state converts them
to ready, partial, sentence_final, final, and error.

Confirmed text never changes. A conflict poisons only the affected session.
Partial replaces the active tail. Sentence_final appends confirmed text.
Exactly one final terminates success, no event follows it, and strict success
requires close code 1000.

## Failure handling and backpressure

Independent limits cover active sessions, WebSocket frames, per-session
unprocessed audio, global queued audio, ready jobs, worker reservations, queue
age, inference deadline, and connection lag.

Rejected reservations leave audio owned by the session until safe retry or
explicit failure. Results are published only after cleanup and reservations are
released.

Whole-batch failure fails every affected job. Per-item preservation is allowed
only when the adapter guarantees isolation. CUDA OOM clears worker readiness.
Stateful work is never silently retried or moved.

Disconnect, cancellation, endpoint, finish, and worker-loss races use explicit
barriers rather than sleeps. Control actions have priority while preserving
accepted sample ownership.

## Observability

Every job records monotonic times for audio_received, chunk_ready,
scheduler_enqueued, scheduler_dispatched, worker_accepted, inference_started,
inference_completed, result_applied, and event_sent.
`result_applied` is recorded only after locked protocol-state application.
`event_sent` is recorded by the outbound owner only after the last real event
for that job is successfully written; disconnected or queued-but-unsent events
do not count as completed jobs.

Metrics include active sessions, buffered and queued audio, ready depth, batch
size and fill ratio, batch wait, worker wait, inference latency, decoded audio,
aggregate RTF, update interval, lag failures, cancellations, readiness, and
result conflicts. Logs contain IDs and measurements, never credentials or raw
audio.

Health means the process is alive. Readiness requires a capable registered
worker with completed real warmup and available admission capacity.

## Configuration

Initial validated fields include:

    ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=20
    ASR_GATEWAY_MAX_READY_JOBS
    ASR_GATEWAY_MAX_QUEUED_AUDIO_SECONDS
    ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS
    ASR_GATEWAY_DEFAULT_UPDATE_MS=1500
    ASR_GATEWAY_DRAIN_TIMEOUT_SECONDS

Worker capabilities define model batch and chunk limits. Gateway configuration
cannot override an advertised safe maximum.

## Repository shape

The implementation replaces the transparent gateway. Focused modules are:

    app/asr_gateway.py
    app/asr_gateway_protocol.py
    app/asr_gateway_sessions.py
    app/asr_gateway_chunking.py
    app/asr_gateway_scheduler.py
    app/asr_gateway_backends.py
    app/asr_gateway_metrics.py
    tests/test_asr_gateway*.py

Fewer files are acceptable when cohesion remains clear. A monolithic module
mixing protocol, PCM ownership, scheduling, and model execution is not.

## Deterministic test strategy

Required tests prove:

1. Authentication, start and frame validation, idle, finish, and terminal
   protocol behavior.
2. PCM conservation through chunks, boundaries, endpoint, finish, cancellation,
   and worker failure.
3. Multiple sessions released by a barrier form one dynamic batch.
4. A serial worker receives only lists of length one with the same scheduler.
5. Timeout dispatch uses a controllable clock or event, not sleeps.
6. Incompatible batch keys never merge.
7. One session cannot have two in-flight jobs or monopolize a round.
8. Stale and out-of-order results fail or are discarded explicitly.
9. Stateful worker loss does not reroute.
10. Retry-safe stateless work reroutes only before acceptance.
11. Every failure releases reservations exactly once.
12. Registry drain and replacement preserve stickiness.
13. Transcript sequence, confirmed prefix, one final, and close 1000 hold.
14. Metrics contain every stage and no raw audio or credentials.
15. Existing services and release verification remain green.

Concurrency proofs use deterministic barriers. Stress loops are supplemental.

## Delivery and external gates

The candidate passes focused Gateway tests and
scripts/verify_asr_release.sh commit, then independent read-only acceptance
against the exact SHA, then fresh primary product regression.

Real image build, A10 warmup, throughput, accuracy corpus, high-concurrency
continuous speech, GPU utilization, VRAM headroom, and soak are external gates.
Missing hardware evidence is unexecuted, never inferred from fake adapters.
