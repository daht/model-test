# Stateful ASR Hardening Design

## Objective

Harden the production `qwen_vllm + stateful` ASR path so that:

- confirmed transcript text is never duplicated, removed, or revised;
- replaceable partial text has explicit and deterministic semantics;
- synchronous Qwen/vLLM work never blocks the FastAPI event loop;
- overload, timeouts, disconnects, and malformed client input fail predictably;
- one GPU is owned by one ASR process with bounded resource usage;
- behavior is verified with unit tests, protocol tests, concurrency tests, and real audio.

Translation and TTS are outside this change. Chunked ASR remains a supported fallback but only receives a narrow correctness fix.

## Production Assumptions

- The primary backend is `ASR_BACKEND=qwen_vllm`.
- The primary stream mode is `ASR_STREAM_MODE=stateful`.
- Input audio is mono, 16 kHz, signed 16-bit little-endian PCM.
- One Uvicorn process owns one GPU model instance.
- Stateful Qwen/vLLM calls are treated as non-thread-safe and non-reentrant.
- Live streaming has priority over batch file transcription.

## Architecture

```text
FastAPI HTTP/WebSocket protocol layer
                 |
                 v
       StreamingSessionController
       - validates protocol
       - tracks session lifecycle
       - translates state events to JSON
                 |
                 v
       StreamingTranscriptState
       - confirmed prefix
       - replaceable partial tail
       - stable punctuation candidate
       - VAD and finalization rules
                 |
                 v
       ASRInferenceCoordinator
       - single model-owner thread
       - bounded deadline-aware queue
       - streaming and batch admission
       - cancellation/poison handling
                 |
                 v
       QwenVLLMASRTranscriber
       - load and warmup
       - init/add/finish/abort session
```

Pure transcript state is separated from model execution. The API layer does not manipulate confirmed or partial strings directly. All model lifecycle calls, including model loading, execute on the coordinator's dedicated thread.

## Transcript Protocol

### Invariants

For each connection:

1. `confirmed_text` is append-only.
2. `partial` is the complete replaceable tail, not a delta.
3. The client-visible transcript is always:

   ```text
   concatenated sentence_final events + latest partial or final tail
   ```

4. A commit emits `sentence_final` followed by the remaining `partial`, including `partial: ""` when no tail remains.
5. `final` contains only the remaining unconfirmed tail and is the last transcript event.
6. Every server event carries a monotonically increasing `sequence` number.
7. Model output that conflicts with an already confirmed prefix never causes confirmed text to be re-emitted as partial text.

### Protocol Version

`/v1/transcribe/stream-info` exposes `protocol_version: 2`. Version 2 adds event sequence numbers, explicit empty partials after commits, and stable error codes. Sequence and error-code fields are additive, and existing clients must tolerate them. The server does not maintain a second version 1 state machine; rollback uses the previous image if a client incompatibility is discovered.

### Stable Punctuation

Stable punctuation uses processed audio time, not server wall-clock time. The state stores cumulative processed samples and derives:

```text
audio_time_seconds = processed_samples / 16000
```

A punctuation candidate becomes committable only when:

- it is an exact prefix of the current unconfirmed tail;
- it meets the minimum non-whitespace character count;
- it survives the configured minimum update count;
- the model has processed the configured amount of additional audio since the candidate first appeared.

Candidate removal, movement, or prefix revision resets stability. VAD force-commit resets the candidate.

### Confirmed-Prefix Conflict

If cumulative model text no longer starts with confirmed text:

- retain confirmed text unchanged;
- derive a tail only from a safe confirmed-suffix/model-prefix overlap;
- never emit the full conflicting model text as a new tail;
- increment a conflict metric and emit a structured warning without transcript content by default;
- close the session with a stable error if no safe continuation can be derived.

Silently duplicating or contradicting confirmed text is not allowed.

## Inference Coordinator

### Model Ownership

The coordinator creates, loads, warms, and calls the model on one dedicated worker thread. Calls from the event loop are represented as queue jobs and resolved through async futures.

### Queue Jobs

Each job contains:

- job type: warmup, init session, stream chunk, finish, abort, or file transcription;
- session identifier when applicable;
- priority and FIFO sequence;
- enqueue time and execution deadline;
- callable and result future;
- cancellation state.

The queue is bounded before submission. A job whose deadline expires before execution is discarded without touching model state.

### Timeout Semantics

- Queue timeout: safe cancellation; the model state was not touched.
- Running timeout: Python cannot stop a running GPU call. Mark the session poisoned, close the connection, and never reuse that state.
- Disconnect during a running call: discard the eventual result and abort/release the session after the running call completes.
- A timed-out or disconnected job may consume capacity until the underlying model call returns, but it cannot mutate a live replacement session.

### Scheduling

Streaming chunks have higher priority than file jobs, but a running file transcription cannot be preempted. Therefore the live production service defaults to:

```dotenv
ASR_FILE_TRANSCRIBE_ENABLED=false
```

Batch transcription should use a separate ASR instance. If batch remains enabled on a shared instance, a file job may begin only when there are no active streams, and new streams are rejected while that non-preemptible batch job is running.

### Capacity and Backpressure

Capacity is bounded by all of the following:

- active streaming sessions;
- queued jobs;
- queued audio seconds globally;
- unprocessed audio seconds per connection;
- binary frame bytes;
- maximum session duration and cumulative audio duration.

When a limit is reached:

- HTTP returns 503 with a stable error code;
- WebSocket sends an error event and closes with code 1013;
- stale audio is never processed indefinitely as if it were real time.

Initial capacity values are conservative and must be calibrated on the A10. The maximum active stream count is not considered valid until benchmarked.

## Session Lifecycle

State transitions are:

```text
accepted -> authenticating -> ready -> active -> finishing -> closed
                                      \-> poisoned -> closed
```

The server enforces:

- start-message timeout;
- idle timeout;
- maximum connection duration;
- maximum cumulative audio duration;
- exactly one in-flight model operation per session;
- cleanup in a `finally` path for normal end, disconnect, protocol error, timeout, and inference error.

The ASR session interface gains an explicit `abort()` or `close()` operation. Direct mutation of the qwen-asr state's internal `buffer` is removed. Segment reset uses an official reset API when available; otherwise it creates a fresh official streaming state while preserving only application-level confirmed transcript state.

## Input Validation

The WebSocket start message is validated as structured data:

- `type` must be `start`;
- API key must be valid;
- `format` must be `pcm_s16le`;
- `sample_rate` must be the integer 16000;
- `language` must be null or a bounded string accepted by the backend;
- unknown fields are ignored for forward compatibility, but invalid known fields are rejected.

Binary frames must be non-empty, even-length PCM and below the configured frame byte limit. Malformed JSON after start produces a protocol error rather than an unhandled exception.

## Health and Readiness

- `/health` is liveness only and remains fast while inference is running.
- `/ready` reports ready only after the coordinator thread is running, model warmup has succeeded, and the service is accepting jobs.
- Warmup failure is retained as sanitized readiness detail and does not expose paths, secrets, or transcript content.
- Production uses eager warmup in application lifespan.
- Shutdown stops admission, drains or expires queued jobs, aborts sessions, and joins the worker within a bounded grace period.

## Configuration

New settings use validated positive bounds:

```dotenv
ASR_PROTOCOL_VERSION=2
ASR_EAGER_LOAD=true
ASR_FILE_TRANSCRIBE_ENABLED=false
ASR_MAX_ACTIVE_STREAMS=2
ASR_INFERENCE_QUEUE_SIZE=16
ASR_MAX_QUEUED_AUDIO_SECONDS=4.0
ASR_MAX_CONNECTION_LAG_SECONDS=2.0
ASR_MAX_FRAME_BYTES=32000
ASR_START_TIMEOUT_SECONDS=10
ASR_IDLE_TIMEOUT_SECONDS=30
ASR_MAX_SESSION_SECONDS=1800
ASR_STREAM_QUEUE_TIMEOUT_SECONDS=2.0
ASR_STREAM_INFERENCE_TIMEOUT_SECONDS=15.0
ASR_FILE_INFERENCE_TIMEOUT_SECONDS=300.0
```

`ASR_MAX_ACTIVE_STREAMS=2` is an intentionally conservative rollout value, not a capacity claim. A10 benchmarks determine the final value.

## Chunked Fallback

Chunked mode transcribes independent, non-overlapping audio buffers. Its outputs are appended as independent segments and are never passed through cumulative-output deduplication. Repeated speech across adjacent chunks must remain repeated in the transcript.

No broader chunked-mode redesign is included.

## Observability and Privacy

Structured logs and metrics cover:

- active sessions;
- queue depth and queued audio seconds;
- queue wait and inference duration;
- partial and commit counts;
- confirmed-prefix conflicts;
- timeout, overload, disconnect, and inference error counts;
- real-time factor when measurable.

API keys, raw audio, and full transcripts are not logged. Transcript logging is off by default and may only be enabled explicitly in a controlled environment.

## Testing

### Pure State Tests

- monotonic confirmed prefix;
- replaceable and empty partial behavior;
- VAD, punctuation, segment, and end transitions;
- punctuation stability based on audio time;
- punctuation revision and disappearance;
- confirmed-prefix conflict handling;
- abbreviations, decimals, domains, and multilingual terminators;
- event sequence monotonicity.

### Coordinator Tests

- model construction and every model call occur on the owner thread;
- per-session ordering;
- bounded admission and queue timeout;
- expired jobs never touch model state;
- running timeout poisons a session;
- disconnect cleanup;
- exception isolation and continued worker operation;
- shutdown behavior;
- streaming priority and batch admission rules.

### API Tests

- invalid start JSON and fields;
- invalid PCM frame size and alignment;
- active-session and lag limits;
- stable HTTP 503 and WebSocket 1013 overload behavior;
- liveness remains responsive during fake slow inference;
- readiness before, during, and after warmup;
- version 2 event reconstruction.

### Real Model Acceptance

Use the available Chinese and Japanese recordings plus labeled short fixtures. Record raw protocol events and verify:

- reconstructed text never duplicates confirmed content;
- confirmed events are never revised;
- the final displayed text matches the model's completed transcript policy;
- first-partial latency, queue wait, inference latency, and real-time factor;
- behavior at 1, 2, 4, and 8 concurrent real-time streams;
- CER on labeled fixtures and manual review on longer recordings.

Capacity settings are updated only from measured A10 results.

## Rollout

1. Land transcript state and protocol tests behind protocol version 2.
2. Land the coordinator with conservative limits and file transcription disabled.
3. Add eager readiness and operational metrics.
4. Deploy to a staging A10 and run single-stream real audio tests.
5. Run controlled concurrency tests and calibrate capacity.
6. Enable production traffic gradually while monitoring lag, conflicts, timeouts, and GPU memory.
7. Confirm all production clients tolerate the additive version 2 fields and explicit empty partial events.

Rollback is configuration-based where possible: disable protocol version 2 client exposure or return to the previous image. Confirmed-prefix and coordinator changes are not partially toggled within one session.

## Non-Goals

- Changing the Qwen3-ASR model or adding a general-purpose LLM.
- Redesigning translation or TTS.
- Implementing neural VAD or semantic endpointing.
- Increasing GPU throughput without measurement.
- Supporting multiple Uvicorn workers on one GPU.
- Building a distributed multi-GPU scheduler in this iteration.

## Agent Execution and Acceptance

Implementation begins only after explicit user approval.

1. A dedicated implementation agent works from an approved implementation plan and commits scoped changes.
2. A separate test agent reviews the specification independently, runs automated and available integration tests, and reports defects without accepting implementation claims at face value.
3. The primary agent reviews both agents' work, resolves remaining issues, and performs product-level regression across ASR upload, WebSocket protocol, scripts, documentation, and deployment configuration.
4. Completion is reported only when required checks pass, real-model checks have either passed or are explicitly identified as blocked by unavailable GPU/model infrastructure, and no known in-scope defect remains.
