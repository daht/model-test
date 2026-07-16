# faster-whisper large-v3 Streaming Backend Design

## Objective

Add a selectable `faster_whisper` backend to the semantic ASR Gateway for an
NVIDIA A10 capacity and multilingual-accuracy evaluation. The initial runtime
contract is `large-v3`, FP16, batch size four, beam size one for rolling
partials, and beam size five for utterance-final decoding. The task is always
multilingual transcription; translation is not exposed.

The existing Qwen backends remain installed and selectable. This change does
not claim a production capacity, CER/WER result, or live readiness until the
same committed image, approved model manifest, A10, and external speech pass
the release and live gates.

## Runtime Architecture

The Gateway remains the only public WebSocket server and continues to own
authentication, Silero VAD, session admission, PCM accounting, deadlines, and
the normalized event protocol. `FasterWhisperAdapter` is a new in-process model
adapter with these capabilities:

- `StreamingMode.ROLLING`
- `DispatchMode.DYNAMIC_MICROBATCH`
- `ResultMode.REPLACEABLE_SEGMENT`
- gateway-owned VAD
- up to four compatible sessions per model call
- one CTranslate2 model owner and one Uvicorn worker per GPU

The adapter keeps one accumulated utterance buffer per session. A normal update
appends newly acknowledged PCM and re-decodes the accumulated utterance with
beam one. The returned text replaces the current segment tail. At a VAD
endpoint, maximum utterance boundary, or end of input, the Gateway marks the
scheduled job final; compatible final jobs form a batch and decode with beam
five. The subsequent segment/final control consumes the cached final text and
clears the adapter utterance without a duplicate model call. An explicit
client `segment` issued between scheduled updates uses a single beam-five
fallback because it does not pass through a cross-session endpoint barrier.

The adapter never combines PCM between sessions. It stacks independent
log-Mel features into one CTranslate2 encoder/decoder batch and maps each output
back to the input job by position and identity.

## Language Behavior

An explicit Whisper language code is fixed for the session. `auto` sessions
are initially decoded with per-item multilingual detection in the batch. The
detected language is then stored in that session and reused for later calls.
If one scheduler batch contains auto sessions that have locked to different
languages, the adapter partitions it into language-compatible sub-batches and
restores the original result order.

Only `task=transcribe` and `timestamps=false` are accepted. This prevents an
accidental Whisper translation task and keeps result semantics identical to the
current text-only WebSocket protocol.

## Model And Provenance

The runtime pins `faster-whisper==1.2.1` and `ctranslate2==4.8.1`, matching the
existing CUDA 12/cuDNN 9 image. Production configuration points to a local
CTranslate2 `large-v3` directory. The existing model manifest verifier checks
the exact local artifact set before model construction when manifest
verification is enabled.

Model acquisition is an operator-controlled staging action. An immutable
upstream revision is downloaded on a trusted host, a manifest is created
outside the model directory, and both artifacts are transferred together.
Runtime startup must not silently download a model when a local production path
was configured.

## Configuration

The selected values are:

```dotenv
ASR_BACKEND=faster_whisper
ASR_STREAM_MODE=rolling
ASR_MODEL_NAME=large-v3
ASR_MODEL_ID=/models/faster-whisper-large-v3
ASR_FASTER_WHISPER_COMPUTE_TYPE=float16
ASR_FASTER_WHISPER_BATCH_SIZE=4
ASR_FASTER_WHISPER_PARTIAL_BEAM_SIZE=1
ASR_FASTER_WHISPER_FINAL_BEAM_SIZE=5
ASR_FASTER_WHISPER_TASK=transcribe
```

Queue, lag, session capacity, and VAD settings remain independent controls and
must be tuned only from A10 evidence. Increasing batch size does not itself
raise admission limits.

## Failure And Cleanup

A whole CTranslate2 batch failure returns a sanitized error for every affected
job through the existing scheduler failure path and clears worker readiness.
No secret or raw backend message is exposed. Cancellation waits until an
accepted model call reaches the existing cleanup barrier. Abort, finish, close,
and failed startup all remove session PCM and cached text. Result count or
identity mismatches fail closed.

## Verification

Deterministic tests cover configuration validation, capability advertisement,
real cross-session batch formation at the adapter boundary, session PCM
isolation, language locking and sub-batching, partial/final beam selection,
final-cache cleanup, ordering, failure sanitization, runtime selection, and
Qwen rollback. The staged candidate must pass `scripts/verify_asr_release.sh
commit` before commit.

External gates on the A10 are model-manifest verification, image build, load and
non-silent warmup, Chinese/English/Japanese/Korean/Yue accuracy samples,
concurrency and latency sweeps, GPU utilization, peak VRAM, soak, and rollback.
