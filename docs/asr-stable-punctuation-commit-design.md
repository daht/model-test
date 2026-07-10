# ASR Stable Punctuation Commit Design

## Goal

Emit useful `sentence_final` events during continuous stateful ASR without immediately committing transient punctuation such as the early `肯德基。` hypothesis.

## Current Behavior

Stateful Qwen ASR continuously returns cumulative, replaceable text. The server currently commits text only when VAD detects a configured duration of silence. Continuous live-commerce speech may not contain 500-1500 ms of silence, so the entire utterance remains one `partial` until trailing silence or `end`.

The official `ASRStreamingState` improves token continuity and allows recent text to be revised, but it does not decide when application-level text is immutable.

## Proposed Behavior

Stateful mode uses two independent commit triggers:

1. VAD commit: commit all pending text after 0.8 seconds of detected silence.
2. Stable punctuation commit: commit a punctuated prefix after the exact boundary has remained unchanged for at least 1.0 second and at least two stateful recognition updates.

Chunked mode keeps its existing behavior.

## Stable Boundary Rules

A punctuation boundary is eligible when all conditions are true:

- The boundary ends with an existing sentence terminator recognized by `SentenceCommitter`.
- The candidate contains at least 8 non-whitespace characters.
- The exact candidate prefix appears unchanged in at least two consecutive stateful updates.
- At least 1.0 second has elapsed since that exact candidate first became stable.
- The candidate is part of the current cumulative ASR text after removing already confirmed text.

When eligible, the server emits one `sentence_final` for the stable prefix and keeps the remaining text as the replaceable `partial` tail.

Example:

```text
update 1: 肯德基。
update 2: 肯德基双人餐八件套兑换券。
update 3: 肯德基双人餐八件套兑换券，限时优惠。
update 4: 肯德基双人餐八件套兑换券，限时优惠。仅需七十九元
```

The first short candidate is rejected by minimum length. If the longer sentence boundary remains unchanged for 1.0 second across at least two updates, the server sends:

```json
{"type":"sentence_final","text":"肯德基双人餐八件套兑换券，限时优惠。"}
{"type":"partial","text":"仅需七十九元"}
```

## Configuration

Add:

```dotenv
ASR_STABLE_COMMIT_ENABLED=true
ASR_STABLE_COMMIT_SECONDS=1.0
ASR_STABLE_COMMIT_MIN_CHARS=8
ASR_STABLE_COMMIT_MIN_UPDATES=2
```

Change the stateful production example to:

```dotenv
ASR_VAD_SILENCE_SECONDS=0.8
```

The generic `.env.example` may keep stable commit enabled because it only affects `ASR_STREAM_MODE=stateful`. Existing chunked deployments remain behaviorally unchanged.

## Components

### StablePunctuationCommitter

Add a focused state object in `app/asr_api.py` or a small ASR streaming helper module. It tracks:

- Current candidate prefix.
- Monotonic time when the candidate first appeared.
- Number of consecutive updates containing the exact candidate.

It accepts the current unconfirmed cumulative tail and returns zero or more stable sentences to commit. A changed or removed candidate resets its timer and update count.

### WebSocket Integration

In `_run_stateful_transcribe_stream`:

1. Apply the cumulative ASR result to `SentenceCommitter` without punctuation auto-commit.
2. Pass the current pending tail to `StablePunctuationCommitter`.
3. Commit eligible stable prefixes through `SentenceCommitter` and emit `sentence_final`.
4. Emit the remaining replaceable `partial`, including an empty partial when a previous tail is cleared.
5. Reset stable-candidate state on VAD commit and `segment`.
6. On `end`, finish official streaming state, run one final stable update, then return the remaining tail through `final`.

The server must never emit the same confirmed prefix twice.

## Time Source

Production uses `time.monotonic()`. Tests inject or pass explicit timestamps so the 1.0-second rule is deterministic and does not sleep.

## Stream Info

Expose the effective stable-commit configuration in `/v1/transcribe/stream-info` under `audio_format.stateful`:

```json
{
  "stable_commit_enabled": true,
  "stable_commit_seconds": 1.0,
  "stable_commit_min_chars": 8,
  "stable_commit_min_updates": 2
}
```

## Script Behavior

`scripts/stream_asr_client.py --print-mode events` continues to display protocol events and should show intermediate `sentence_final` messages.

`--print-mode display` continues to render all confirmed text plus the latest replaceable tail. No client protocol changes are required.

`scripts/smoke_asr.sh` may optionally validate `EXPECT_ASR_STABLE_COMMIT_ENABLED` from stream-info.

## Error And Edge Cases

- If punctuation disappears or moves, reset the candidate instead of committing it.
- If ASR revises text before a candidate boundary, reset the candidate.
- If multiple stable sentences exist, commit them in order up to the last eligible stable boundary.
- Do not confirm a candidate shorter than the configured minimum.
- VAD commit takes precedence and resets stable tracking.
- Empty cumulative text clears the displayed partial and resets stable tracking.
- `segment` discards pending official audio buffer and resets stable tracking without re-emitting confirmed text.

## Testing

Required automated coverage:

- Transient short `肯德基。` is not committed.
- A long punctuation boundary is not committed before 1.0 second.
- The same boundary is committed after 1.0 second and at least two updates.
- A revised or removed boundary resets stability.
- Stable commit emits only the confirmed prefix and leaves the correct partial tail.
- Multiple eligible sentences are committed in order without duplication.
- VAD and `segment` reset stable tracking.
- Chunked mode tests remain unchanged.
- Stream-info and script configuration checks include stable-commit settings.

## Rollout

Deploy behind `ASR_STABLE_COMMIT_ENABLED=true`. Validate with the existing live-commerce recording and inspect raw event mode. Success means several semantically useful `sentence_final` events appear during continuous speech, while the transient initial `肯德基。` hypothesis remains a replaceable partial.

## Non-Goals

- No neural semantic endpointing model in this iteration.
- No timestamp or word-alignment support.
- No change to official `ASRStreamingState` internals.
- No immediate punctuation commit on the first appearance of a terminator.
