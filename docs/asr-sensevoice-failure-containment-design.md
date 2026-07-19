# SenseVoice Batch Failure Diagnosis and Containment Design

## Objective

Make the first SenseVoice batch failure diagnosable without exposing model output,
audio, credentials, or raw exception text, and contain that failure so it cannot
cause further dispatch, duplicate terminal events, or cleanup/result conflicts.

The July 19 A10 capacity run is the acceptance scenario: 80 synchronized streams
passed, while 88 streams produced one dispatched batch of eight jobs with no
matching engine-completion event. Four sessions received adapter errors, four
expired on connection lag, and four cleanup conflicts followed. The failure was
not caused by the 320-second queue limit, VRAM exhaustion, or an inference
timeout.

## Non-goals

- Do not change admission limits, queue limits, timeouts, rolling-window size, or
  SenseVoice batch size.
- Do not retry a failed batch or automatically reduce its batch size. The local
  SenseVoice backend is not retry-safe.
- Do not claim that 88 streams will pass after this change. A new live run must
  identify the first failure stage before any model/runtime correction.
- Do not log raw exception messages, model results, transcripts, audio, or
  credentials.

## Design

### Stable failure classification

SenseVoice batch execution will distinguish four stable stages:

- `engine_generate`: the FunASR model call raised;
- `result_contract`: FunASR returned a non-list or malformed item;
- `result_count`: the number of returned items differed from the number of
  inputs;
- `result_omitted`: the adapter could not map a decoded item back to a submitted
  job.

The adapter will emit one `asr_engine_group_failed` structured event before
propagating the error. Safe fields are the stage, exception type, worker and
batch identity, group size, group ordinal/count, final-item count, accumulated
audio seconds, and minimum/maximum input audio seconds. The event will not
contain exception text or model output.

### Worker fail-stop

The scheduler will treat the first adapter submission exception as fatal for
that worker. It will stop selecting that worker for subsequent dispatch and
report `submit_failed` exactly once. Current accepted jobs and queued jobs must
have their queue/sample accounting and reservations settled before failure is
published.

Queued work for the failed worker will not be retried or sent to the adapter.
Each affected job will receive a sanitized failure result only after its
ownership cleanup is complete. The scheduler must finish with zero queued jobs
and zero queued samples for that worker.

### Idempotent terminal handling

A session may emit exactly one `asr_session_terminal` event. Failure handling
must not abort an adapter session until scheduler ownership for that session is
safe. Results arriving for an already terminal or stale generation are discarded
without another failure transition or terminal event.

The result-error path and external `enqueue_error` path must share the same
terminal invariants even though scheduler publication cannot wait on its own
accepted job. The scheduler therefore settles accepted/queued ownership before
publishing fatal results, and the gateway performs an idempotent terminal
transition after that boundary.

## Deterministic verification

Tests will use a controlled adapter whose first submitted batch fails while more
jobs are queued. They must prove:

- the test fails against the current implementation;
- only one adapter submission occurs after the injected fatal failure;
- the worker is marked failed once;
- no queued job remains and queued-sample accounting returns to zero;
- each affected session emits exactly one terminal event;
- no cleanup conflict or result conflict occurs;
- every session, backend session, reservation, and lease is released;
- the structured SenseVoice failure event contains the required safe fields and
  excludes raw exception text and model output.

Focused SenseVoice, scheduler, gateway, monitoring-report, configuration, and
deployment tests must pass, followed by the repository commit gate. Independent
read-only acceptance must verify the committed SHA before product regression.

## Rollout and live verification

After local acceptance, deploy the same accepted SHA with the existing A10
SenseVoice configuration. Recreate the container before every capacity tier.
Run one monitored 88-stream test first. If it fails, the first
`asr_engine_group_failed` event must identify the stage without any downstream
dispatch or lifecycle conflicts. Only after correcting that identified root
cause should the 80/88/92/96 capacity sweep resume.

## Risks

Fail-stop reduces availability after one fatal batch, but continuing to use a
backend whose state is unknown produced substantially larger failure cascades in
the observed run. Failing closed is the required behavior until the underlying
FunASR or result-contract failure is understood. Incorrect cleanup ordering can
deadlock scheduler publication, so deterministic tests must exercise both the
current accepted batch and queued work behind it without sleeps.
