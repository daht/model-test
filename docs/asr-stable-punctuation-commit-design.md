# Stateful Punctuation Commit Design (Superseded)

This document records a retired design. Stateful ASR no longer treats punctuation or unchanged text duration as evidence that a model hypothesis is immutable.

The compatibility settings `ASR_COMMIT_ON_PUNCTUATION` and `ASR_STABLE_COMMIT_*` are still accepted so older deployments do not fail configuration parsing. Their effective values are always false when `ASR_STREAM_MODE=stateful`, and `/v1/transcribe/stream-info` reports that explicitly.

The production contract is now:

- Qwen returns a replaceable, segment-local snapshot.
- Silero VAD v6.2.1 on ONNX Runtime CPU controls speech admission and normal acoustic endpoints.
- Explicit `segment` and the 30-second normal utterance limit also flush and finalize the current segment.
- The 120-second watchdog is an invariant abort and is not a normal rollover.
- `end` flushes the last state and sends its still-unconfirmed text only in `final`.
- Clients reconstruct the transcript as ordered `sentence_final` values plus the single terminal `final` value.

See `docs/API.md` and `README.md` for the active configuration and protocol.
