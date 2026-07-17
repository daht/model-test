# Faster-Whisper Batch Resynchronization Design

Date: 2026-07-17

## Objective

Keep compatible `faster_whisper` rolling sessions inside one scheduler
submission across VAD boundaries so that endpoint jitter cannot permanently
split an initially healthy five-stream batch. Restore diagnostic INFO events so
the same live workload can prove scheduler and engine behavior.

## Evidence

The five-stream A10 run started with nine consecutive five-item engine calls.
At the first VAD boundary, scheduler fragmentation simultaneously reported
`partial_final_identity` and `length_bucket`; the next work split into four-item
and singleton calls. The split persisted, GPU utilization reached 99 percent,
partial inference reached about two seconds, and two sessions hit the exact
six-second `session_pcm_limit`.

The adapter already partitions one scheduler submission by language and beam,
but live `groups_per_scheduler_batch` remained exactly one. The gateway's
stricter `BatchKey` therefore prevented the adapter grouping boundary from ever
receiving mixed partial/final work. HY-MT was idle, total GPU memory remained
below half of the A10 capacity, and cleanup returned all gauges to zero.

The first resynchronization candidate still failed live after 106 seconds. Its
last two calls were complete five-item batches, but decoding 26.8 and 28.8
seconds per session took 2.179 and 2.279 seconds. That exceeds the two-second
input cadence even with optimal batching, proving that repeated full-segment
rolling decode is the primary capacity limit. Batch fragmentation is a
secondary amplifier.

The 15-second candidate removed that sustained growth but exposed a second,
independent decoder bound after about 223 seconds. Of 195 engine calls, 192
normal calls averaged 0.601 seconds and never exceeded 1.586 seconds. The three
abnormal calls each contained a 224-character run, averaged 2.378 seconds, and
immediately preceded the session buffer rejection. The repeated output occurred
with only 8–14 seconds of accumulated audio, so shortening the rolling window
alone cannot prevent it.

The monitor also captured only warning-level slow calls despite
`ASR_DIAGNOSTIC_LOGGING=true`. Setting the dedicated logger level was
insufficient because it had no production handler.

## Selected Design

For `faster_whisper` only, construct scheduler compatibility from the decoding
options shared by the adapter submission. Do not encode partial/final identity
or transport chunk length in its `BatchKey`. The adapter remains responsible
for its existing language/beam partitioning, including separate beam-one and
beam-five engine groups within one scheduler submission. Because `submit()`
returns only after all internal groups finish, result cleanup and publication
again form one synchronization barrier for the sessions in that submission.

All other backends retain their current partial/final and length-bucket keys.
The A10 faster-whisper contract uses a 15-second maximum utterance boundary so
the adapter clears its accumulated PCM before full-batch inference exceeds the
two-second arrival cadence. Buffer, timeout, VAD, beam, and protocol semantics
remain unchanged; continuous speech emits confirmed segments more frequently.
The engine also sets `no_repeat_ngram_size=3` to prevent the observed repeated
decoder sequence from consuming the whole batch budget. It does not impose an
untested `max_new_tokens` truncation.

Emit through a child of the configured `uvicorn.error` logger so production
handlers receive INFO records.
Diagnostic gating still suppresses high-frequency events unless
`ASR_DIAGNOSTIC_LOGGING=true`; always-on lifecycle events remain visible.

## Failure Handling

Queue limits, session PCM limits, inference timeouts, cancellation barriers,
generation checks, and terminal cleanup remain unchanged. A mixed adapter
submission still fails as one sanitized scheduler batch if either internal
engine group fails.

## Verification

Add deterministic regression coverage proving that:

1. Four final jobs and one partial job from compatible faster-whisper sessions
   enter one scheduler submission and become two internal beam groups.
2. Different boundary chunk lengths do not split faster-whisper jobs.
3. Other backends still separate final/partial and length buckets.
4. Diagnostic INFO events are emitted through normal logging when enabled and
   remain gated when disabled.

Run focused gateway, scheduler, faster-whisper, and observability tests, then the
full explicit-mock suite and `scripts/verify_asr_release.sh commit`. The live
acceptance test is the same five-stream real-time workload with HY-MT resident;
it must complete without capacity rejection and must show mixed scheduler
batches, restored full batches after boundaries, and zero accounting gauges
after cleanup.
