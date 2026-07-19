# SenseVoice Small Streaming Backend Design

Date: 2026-07-19

## Objective

Add a selectable `sensevoice` backend to the semantic ASR Gateway so the same
WebSocket client, VAD, scheduler, protocol, and A10 monitoring workflow can
compare SenseVoice Small with the existing Qwen3-ASR and faster-whisper
backends. The evaluation measures recognition behavior, sustainable real-time
concurrency, latency, and the first saturated resource on one NVIDIA A10.

The existing Qwen and faster-whisper backends remain selectable rollback paths.
This change does not replace a production backend or claim capacity, accuracy,
release readiness, or live readiness before the matching committed image,
model manifest, A10 deployment, and external speech pass their applicable
gates.

## Non-goals

- Implement native stateful streaming that SenseVoice Small does not expose.
- Share a speculative base class with `FasterWhisperAdapter`.
- Change the public client commands, audio format, authentication, VAD policy,
  admission policy, or confirmed-text rules.
- Download model weights during container startup.
- Load SenseVoice and another ASR model on the same GPU simultaneously.
- Add timestamps, translation, hotwords, diarization, or arbitrary FunASR
  pipeline components to this evaluation.

## Selected Approach

The selected implementation is an independent in-process
`SenseVoiceAdapter`, using FunASR on PyTorch CUDA and Gateway-owned VAD. It
advertises:

- `StreamingMode.ROLLING`
- `DispatchMode.DYNAMIC_MICROBATCH`
- `ResultMode.REPLACEABLE_SEGMENT`
- gateway-owned VAD
- multilingual transcription for `zh`, `yue`, `en`, `ja`, and `ko`

The independent adapter is preferred over a shared faster-whisper base class
because the runtimes have different feature extraction, batch APIs, return
formats, tag handling, and cleanup behavior. It is preferred over a separate
service because the current single-GPU evaluation already has a bounded
in-process adapter/scheduler contract; adding another network and process
boundary would introduce unrelated latency and failure variables.

## Rolling Recognition Flow

Each backend session owns one accumulated, unconfirmed utterance buffer. The
Gateway acknowledges newly consumed PCM exactly once. A scheduled rolling
update appends that new PCM, re-decodes the complete current utterance, and
returns a replaceable segment snapshot. Consequently, a later `partial`
replaces the earlier partial rather than appending duplicate text.

At a Gateway VAD endpoint, exact maximum-utterance boundary, explicit client
`segment`, or `finish`, the adapter performs a final decode of the accumulated
utterance. The Gateway then commits it as `sentence_final`; `finish` also emits
the single terminal `final`. The finalized adapter buffer is cleared only after
the result has crossed the existing scheduler cleanup barrier. Audio from
different sessions is never concatenated; compatible items share one model
batch and results map back to jobs by input position and identity.

This is pseudo-streaming. It intentionally trades repeated computation for the
existing low-latency WebSocket behavior while keeping SenseVoice's offline
model result replaceable until an official endpoint commits it.

## Text And Metadata Normalization

SenseVoice control tags are removed from user-visible text. Recognized language,
emotion, and audio event are normalized into an optional JSON-safe metadata
object:

```json
{
  "metadata": {
    "language": "zh",
    "emotion": "neutral",
    "audio_event": "speech"
  }
}
```

`AdapterResult` and `InferenceResult` gain an optional metadata mapping, and
the Gateway passes it to protocol serialization without interpreting
backend-specific values. Only events produced from a result carrying metadata
receive this field. A session retains the most recent metadata for its current
unconfirmed segment so the corresponding `sentence_final` and terminal
`final` do not lose it. Clearing or rolling over that segment also clears its
metadata.

Qwen, faster-whisper, and mock payloads remain byte-for-byte compatible at the
schema level: when metadata is absent, no `metadata` key is serialized.
Unknown, malformed, or missing tags do not fail otherwise valid recognition;
they are omitted from metadata. Model output must not be copied into metadata.

## Model, Runtime, And Warmup

The runtime pins `funasr==1.3.14`. The model is staged separately on the server
at `/models/SenseVoiceSmall`, with the approved manifest at
`/models/SenseVoiceSmall.manifest.json`. The adapter loads only the configured
local directory. A missing directory, invalid manifest, unavailable CUDA
runtime, incompatible FunASR version, or model construction failure keeps
readiness false and returns a sanitized startup error.

Warmup runs after manifest verification and model construction. It uses a real,
short, non-silent 16 kHz mono sample through the same preprocessing and model
call used by live jobs. Warmup must return normalized text and exercise tag
parsing; silence or a mocked decode cannot establish readiness.

## Configuration

The initial evaluation configuration is:

```dotenv
ASR_BACKEND=sensevoice
ASR_STREAM_MODE=rolling
ASR_MODEL_ID=/models/SenseVoiceSmall
ASR_MODEL_MANIFEST_PATH=/models/SenseVoiceSmall.manifest.json
ASR_SENSEVOICE_BATCH_SIZE=8
ASR_SENSEVOICE_USE_ITN=true
```

Batch size eight is a conservative starting limit, not an A10 capacity claim.
Existing queue, lag, VAD, utterance-length, connection, and session limits stay
independent. They may be changed for a later sweep only from recorded evidence;
raising the model batch size alone must not silently raise admission limits.

## Failure, Cancellation, And Cleanup

A result-count or result-identity mismatch fails the affected batch closed. A
whole FunASR or CUDA call failure returns sanitized errors for all affected
jobs and clears backend readiness when continuing could be unsafe. No model
path detail, raw backend exception, credential, PCM, or transcript is included
in public errors.

Accepted work observes the existing cleanup barrier: job publication occurs
only after adapter cleanup, reservation release, and queue accounting finish.
Abort, disconnect, timeout, explicit finish, endpoint rollover, failed warmup,
and runtime close remove session PCM, cached text, cached metadata, and backend
session state. Cancellation does not reuse a buffer while its accepted model
call can still access it.

## Deterministic Verification

Repository tests cover:

- configuration validation and `sensevoice` runtime selection;
- capability advertisement and local-model/manifest enforcement;
- tag stripping and normalized optional metadata;
- batch formation, result ordering, result-count validation, and session
  isolation;
- rolling partial replacement and final-decode selection;
- VAD endpoint, explicit segment, finish, `sentence_final`, and terminal `final`
  metadata behavior;
- buffer and metadata cleanup after success, cancellation, disconnect, failure,
  and close;
- non-silent warmup success and fail-closed readiness;
- unchanged Qwen, faster-whisper, mock, protocol sequence, and cleanup behavior.

The intended files are staged explicitly and must pass
`scripts/verify_asr_release.sh commit` before the implementation commit. A
separate read-only acceptance verifies the exact committed SHA, followed by a
fresh product regression.

## A10 Evaluation

After local deterministic acceptance, the server stages the exact model and
manifest, builds the committed image, and passes model load plus non-silent
warmup. The existing external WebSocket load generator sends the same real-user
request path used in earlier Qwen tests; credentials enter only through the
environment. The existing bottleneck monitor records the server side.

The sweep runs at `1`, `8`, `16`, `24`, `32`, and `64` concurrent real-time
streams, stopping escalation when error rate, queue age, connection lag,
completion latency, or resource saturation breaches the predeclared test
threshold. The report records effective real-time factor, end-to-end partial
and final latency, throughput, failures, GPU utilization, VRAM, CPU, memory,
scheduler queueing, and cleanup state. A separate multilingual sample compares
recognition quality; throughput alone is not an acceptable replacement result.

The conclusion distinguishes model inference saturation from scheduler,
preprocessing, CPU, network, VAD, client pacing, or configured admission limits.
Rollback selects the prior backend and its matching model, manifest, image, and
configuration atomically.
