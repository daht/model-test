# Faster-Whisper Batch Coalescing Design

Date: 2026-07-16

## Objective

Restore reliable multi-session streaming for the `faster_whisper` backend while
HY-MT remains resident on the same A10. Compatible real-time streams must form
dynamic microbatches instead of running as repeated singleton inference calls,
and bounded buffering must absorb normal network and scheduling jitter without
hiding sustained lag.

## Evidence

The deployed candidate used a four-item batch capacity, a 20 ms scheduler wait,
a 2,000 ms update interval, and a four-second per-session PCM buffer. External
real-time tests produced these repeatable results:

- one stream completed a 222.72-second recording;
- two streams produced one success and one `invalid_audio`/`BufferError`;
- four streams left one survivor and failed three streams;
- eight streams left one survivor and failed seven streams.

During the deterministic two-stream reproduction, both sessions were active,
but `completed_jobs` and `decoded_seconds` advanced by only one job and two
seconds at a time. `ready_depth` alternated between zero and one,
`queued_audio_seconds` alternated between zero and two, and inference samples
reached 99-100 percent GPU utilization. At failure, one session still had a
two-second reservation and two queued seconds; the next transport frame crossed
the four-second `PcmRingBuffer` limit. The effective jobs were singletons, so
the rolling workload consumed GPU time without the configured cross-session
batch benefit.

## Non-Goals

- Do not change the model, compute type, VAD thresholds, protocol text semantics,
  two-second normal update interval, HY-MT placement, or API authentication.
- Do not make buffering unbounded or weaken the four-second connection-lag and
  eight-second undecoded-age failure limits.
- Do not claim a target concurrency from unit tests. Capacity remains a live A10
  acceptance result.
- Do not solve the separate host Python 3.10 release-runner mismatch or the
  gateway `stream-info`/legacy smoke parser mismatch in this change.

## Considered Approaches

### 1. Balanced coalescing and bounded jitter headroom

Increase the production faster-whisper coalescing window from 20 ms to 200 ms
and the per-session PCM buffer from four seconds to six seconds. Keep the
two-second update interval and all lag deadlines unchanged. Add buffer gauges
and deterministic offset-arrival tests. This adds at most about 200 ms before a
partial batch dispatch and gives one extra update interval of storage without
turning the buffer into a backlog policy.

This is the selected approach because the live trace shows compatible jobs
missing a very small aggregation window, while sustained lag is already bounded
by stricter connection and undecoded-age limits.

### 2. Fixed 500 ms batch cadence

Dispatch on a global fixed cadence. This would align more clients and favor
throughput, but it adds more visible partial latency and requires a larger
scheduler behavior change before the simpler hypothesis has been tested.

### 3. Buffer-only increase

Increase the PCM buffer without improving batch formation. This only delays the
same failure and is rejected.

## Design

### Production tuning contract

The A10 faster-whisper candidate uses:

- `ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=200`;
- `ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=6.0`;
- `ASR_GATEWAY_DEFAULT_UPDATE_MS=2000` unchanged;
- `ASR_MAX_CONNECTION_LAG_SECONDS=4.0` unchanged;
- `ASR_MAX_UNDECODED_AGE_SECONDS=8.0` unchanged;
- `ASR_FASTER_WHISPER_BATCH_SIZE=4` unchanged.

The release verifier checks this backend-specific contract. Qwen and mock
defaults remain unchanged.

### Scheduler behavior

The existing dynamic scheduler remains deadline bounded. The first compatible
job opens a 200 ms collection window; compatible jobs that arrive in that
window share one adapter submission, up to the advertised four-item capacity.
An incomplete batch still dispatches at the deadline, so one-stream traffic
does not wait indefinitely.

The first implementation does not add a fixed global cadence or an active
session barrier. If the real two-stream validation still dispatches singletons,
the captured metrics become evidence for a second design iteration rather than
silently increasing the wait again.

### Buffer bounds

The six-second PCM buffer covers one two-second reservation, one normal queued
update, and one additional update interval of jitter. It does not authorize six
seconds of user-visible lag: the existing four-second connection-lag and
eight-second undecoded-age checks continue to fail sustained overload.

### Observability

The gateway metrics response adds only numeric, aggregate values:

- current total session buffered audio seconds;
- current total session reserved audio seconds;
- current maximum audio seconds held by any session;
- process-lifetime maximum audio seconds held by one session.

No session identifiers, audio, transcript text, prompts, or credentials are
included. Gauges update after ingest scheduling, job cleanup, and session
release so a ring-buffer approach to its bound is visible before cleanup resets
the session.

## Error Handling

Existing safety behavior stays fail-closed. Queue deadlines, audio-lag limits,
buffer limits, adapter failures, and stale result conflicts retain their current
terminal protocol behavior. The repair prevents normal compatible traffic from
reaching the buffer bound; it does not convert overflow into silent dropping or
retry work that the adapter does not declare safe.

## Test Strategy

1. Add a production-contract test that fails against the current 20 ms/four
   second faster-whisper candidate values.
2. Add a deterministic scheduler test with compatible jobs arriving 100 ms
   apart. The first must remain pending and both must be submitted together by
   the 200 ms deadline.
3. Add metric tests for buffered, reserved, maximum-session, and high-water
   values, including reset of current gauges without reset of the high-water.
4. Add a gateway integration test that simulates offset two-session rolling
   updates across repeated cycles and proves reservations are acknowledged
   without crossing the bounded session buffer.
5. Run focused tests, the full explicit-mock suite, and
   `scripts/verify_asr_release.sh commit` against staged files.
6. Deploy the accepted candidate to the test A10 with HY-MT resident. Re-run
   two streams first, then `4/8/12/14/16`, recording success, explicit errors,
   batch fill, latency, GPU use, VRAM, and buffer high-water.

## Acceptance

- Repository baseline and release-contract tests pass on the committed SHA.
- Two real-time streams complete with no BufferError and show batch submissions
  above singleton fill.
- Higher concurrency is reported only from the live sweep; the first failing
  level defines the evaluated capacity.
- Active sessions, ready depth, queued audio, and current session buffer gauges
  return to zero after each stage.
- The Qwen environment and image remain available for immediate rollback.
