# Stateful ASR Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requires one dedicated implementation agent, followed by a separate test agent and final primary-agent product regression.

**Goal:** Make the `qwen_vllm + stateful` ASR service deterministic at the transcript protocol boundary, non-blocking at the FastAPI boundary, bounded under overload, and measurable on an NVIDIA A10.

**Architecture:** Move pure transcript/VAD/commit behavior into `app/asr_streaming.py`. Put the Qwen model and all official streaming session objects behind a single-owner `ASRInferenceCoordinator` in `app/asr_inference.py`; FastAPI talks to it asynchronously and never calls synchronous GPU methods directly. Keep file transcription disabled by default in the live service and retain chunked mode only as a narrow fallback.

**Tech Stack:** Python 3.12, FastAPI, Pydantic Settings, Qwen3-ASR/qwen-asr vLLM, asyncio, `threading.Thread`, `queue.PriorityQueue`, pytest, FastAPI TestClient/WebSocket tests, Docker Compose.

**Design source:** `docs/superpowers/specs/2026-07-11-asr-stateful-hardening-design.md`

---

## File Map

**Create**

- `app/asr_streaming.py`: pure transcript events, confirmed/partial state, audio-time stable punctuation, RMS VAD, punctuation parsing.
- `app/asr_inference.py`: model-owner thread, bounded deadline-aware queue, session registry, async coordinator API.
- `tests/test_asr_streaming.py`: deterministic pure state tests.
- `tests/test_asr_inference.py`: coordinator ownership, ordering, timeout, overload, poison, shutdown, and priority tests.

**Modify**

- `app/asr.py`: warmup/abort lifecycle and safe stateful segment reset without exposing official state to the API thread.
- `app/asr_api.py`: lifespan, dependency injection, strict protocol parsing, coordinator calls, readiness, error mapping, and chunked fallback integration.
- `app/config.py`: validated ASR capacity, lifecycle, readiness, protocol, and frame settings.
- `app/schemas.py`: readiness response and versioned stream-info shape.
- `tests/test_asr_api.py`: protocol v2, validation, overload, readiness, responsiveness, and integration tests.
- `tests/test_asr_api.py`: preserve existing stable punctuation coverage through the extracted state API.
- `tests/test_stream_asr_client.py`: sequence and explicit empty-partial display behavior.
- `scripts/stream_asr_client.py`: protocol version display, event sequence validation, and error-code output.
- `scripts/smoke_asr.sh`: check `/ready` and protocol version.
- `.env.example`: conservative generic defaults.
- `cloud/A10.env.example`: conservative A10 rollout defaults.
- `README.md`, `docs/API.md`, `cloud/README-A10.md`: protocol v2, one-worker rule, file/stream isolation, tuning and rollback.
- `docker-compose.yml`: explicit one-worker ASR command, readiness healthcheck, WebSocket size/queue bounds.

**Do not modify**

- Translation model/API files.
- TTS model/API files.
- Untracked recordings or `docs/million-user-cost-report.html`.

---

### Task 1: Add Validated ASR Runtime Configuration

**Files:**
- Modify: `app/config.py:1-53`
- Create: `tests/test_asr_config.py`
- Modify: `.env.example`
- Modify: `cloud/A10.env.example`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_asr_config.py`:

```python
import pytest
from pydantic import ValidationError

from app.config import Settings


def test_asr_hardening_defaults_are_conservative():
    settings = Settings(_env_file=None)

    assert settings.asr_protocol_version == 2
    assert settings.asr_eager_load is True
    assert settings.asr_file_transcribe_enabled is False
    assert settings.asr_max_active_streams == 2
    assert settings.asr_inference_queue_size == 16
    assert settings.asr_max_frame_bytes == 32000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("asr_max_active_streams", 0),
        ("asr_inference_queue_size", 0),
        ("asr_max_queued_audio_seconds", 0),
        ("asr_max_connection_lag_seconds", 0),
        ("asr_max_frame_bytes", 0),
        ("asr_start_timeout_seconds", 0),
        ("asr_idle_timeout_seconds", 0),
        ("asr_max_session_seconds", 0),
        ("asr_max_audio_seconds", 0),
        ("asr_stream_queue_timeout_seconds", 0),
        ("asr_stream_inference_timeout_seconds", 0),
    ],
)
def test_asr_hardening_settings_must_be_positive(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_pcm_frame_limit_must_be_even():
    with pytest.raises(ValidationError, match="even"):
        Settings(_env_file=None, asr_max_frame_bytes=31999)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_asr_config.py
```

Expected: failures report missing settings.

- [ ] **Step 3: Add bounded settings**

Update `app/config.py` imports and fields:

```python
from pydantic import Field, field_validator

# Inside Settings
asr_protocol_version: Literal[2] = 2
asr_eager_load: bool = True
asr_file_transcribe_enabled: bool = False
asr_max_active_streams: int = Field(default=2, gt=0, le=64)
asr_inference_queue_size: int = Field(default=16, gt=0, le=1024)
asr_max_queued_audio_seconds: float = Field(default=4.0, gt=0, le=120)
asr_max_connection_lag_seconds: float = Field(default=2.0, gt=0, le=30)
asr_max_frame_bytes: int = Field(default=32000, gt=0, le=1_048_576)
asr_start_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
asr_idle_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
asr_max_session_seconds: float = Field(default=1800.0, gt=0, le=86400)
asr_max_audio_seconds: float = Field(default=1800.0, gt=0, le=86400)
asr_stream_queue_timeout_seconds: float = Field(default=2.0, gt=0, le=60)
asr_stream_inference_timeout_seconds: float = Field(default=15.0, gt=0, le=300)
asr_file_inference_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)

@field_validator("asr_max_frame_bytes")
@classmethod
def require_even_pcm_frame_limit(cls, value: int) -> int:
    if value % 2:
        raise ValueError("asr_max_frame_bytes must be even for pcm_s16le")
    return value
```

Add matching variables to both env examples. Keep `ASR_MAX_ACTIVE_STREAMS=2` until A10 measurements justify a change.

- [ ] **Step 4: Run focused tests**

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_asr_config.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_asr_config.py .env.example cloud/A10.env.example
git commit -m "feat: validate stateful asr runtime limits"
```

---

### Task 2: Extract Deterministic Transcript State

**Files:**
- Create: `app/asr_streaming.py`
- Create: `tests/test_asr_streaming.py`
- Modify later: `app/asr_api.py`

- [ ] **Step 1: Write transcript invariant tests**

Create `tests/test_asr_streaming.py` with these initial tests:

```python
import pytest

from app.asr_streaming import ConfirmedPrefixConflict, StreamingTranscriptState


def event_pairs(events):
    return [(event.type, event.text) for event in events]


def new_state(**overrides):
    values = {
        "sample_rate": 16000,
        "stable_commit_enabled": True,
        "stable_commit_seconds": 1.0,
        "stable_commit_min_chars": 8,
        "stable_commit_min_updates": 2,
    }
    values.update(overrides)
    return StreamingTranscriptState(**values)


def test_partial_replaces_previous_partial():
    state = new_state(stable_commit_enabled=False)

    assert event_pairs(state.apply_model_update("hello", processed_samples=1600)) == [
        ("partial", "hello")
    ]
    assert event_pairs(state.apply_model_update("hello world", processed_samples=1600)) == [
        ("partial", "hello world")
    ]


def test_stable_commit_uses_processed_audio_time_and_emits_empty_partial():
    state = new_state()
    text = "这是一个足够长的稳定句子。"

    state.apply_model_update(text, processed_samples=0)
    assert state.apply_model_update(text, processed_samples=8000) == []
    events = state.apply_model_update(text, processed_samples=8000)

    assert event_pairs(events) == [
        ("sentence_final", text),
        ("partial", ""),
    ]
    assert state.confirmed_text == text


def test_wall_clock_delay_without_processed_audio_does_not_commit():
    state = new_state()
    text = "这是一个足够长的稳定句子。"

    state.apply_model_update(text, processed_samples=0)
    assert state.apply_model_update(text, processed_samples=0) == []


def test_vad_commit_emits_sentence_and_empty_partial():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("hello world", processed_samples=1600)

    assert event_pairs(state.commit_pending()) == [
        ("sentence_final", "hello world"),
        ("partial", ""),
    ]


def test_confirmed_prefix_conflict_never_reemits_full_model_text():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("confirmed", processed_samples=1600)
    state.commit_pending()

    with pytest.raises(ConfirmedPrefixConflict):
        state.apply_model_update("different revision", processed_samples=1600)


def test_finish_emits_only_remaining_tail():
    state = new_state(stable_commit_enabled=False)
    state.apply_model_update("hello", processed_samples=1600)

    assert event_pairs(state.finish("hello world")) == [
        ("partial", "hello world"),
        ("final", "hello world"),
    ]


def test_sequences_increase_across_all_events():
    state = new_state(stable_commit_enabled=False)
    first = state.apply_model_update("hello", processed_samples=1600)
    second = state.commit_pending()

    assert [event.sequence for event in first + second] == [1, 2, 3]
```

- [ ] **Step 2: Run tests and verify import failure**

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_asr_streaming.py
```

Expected: collection fails because `app.asr_streaming` does not exist.

- [ ] **Step 3: Implement the public state API**

Create `app/asr_streaming.py` with these public types and signatures:

```python
from __future__ import annotations

from dataclasses import dataclass


class ConfirmedPrefixConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptEvent:
    type: str
    text: str
    sequence: int


@dataclass
class StablePunctuationTracker:
    enabled: bool
    stable_seconds: float
    min_chars: int
    min_updates: int
    candidate: str = ""
    candidate_audio_time: float = 0.0
    candidate_updates: int = 0

    def observe(self, text: str, audio_time: float) -> str | None:
        next_candidate = first_punctuation_candidate(text, self.min_chars) if self.enabled else ""
        if not next_candidate:
            self.reset()
            return None
        if next_candidate != self.candidate:
            self.candidate = next_candidate
            self.candidate_audio_time = audio_time
            self.candidate_updates = 1
            return None
        self.candidate_updates += 1
        if self.candidate_updates < self.min_updates:
            return None
        if audio_time - self.candidate_audio_time < self.stable_seconds:
            return None
        result = self.candidate
        self.reset()
        return result

    def reset(self) -> None:
        self.candidate = ""
        self.candidate_audio_time = 0.0
        self.candidate_updates = 0


class StreamingTranscriptState:
    def __init__(
        self,
        *,
        sample_rate: int,
        stable_commit_enabled: bool,
        stable_commit_seconds: float,
        stable_commit_min_chars: int,
        stable_commit_min_updates: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.confirmed_text = ""
        self.partial_text = ""
        self.processed_samples = 0
        self._sequence = 0
        self.stable = StablePunctuationTracker(
            stable_commit_enabled,
            stable_commit_seconds,
            stable_commit_min_chars,
            stable_commit_min_updates,
        )

    @property
    def audio_time(self) -> float:
        return self.processed_samples / self.sample_rate

    def apply_model_update(self, text: str, *, processed_samples: int) -> list[TranscriptEvent]:
        self.processed_samples += processed_samples
        previous = self.partial_text
        self.partial_text = self._unconfirmed_tail(text)
        stable_prefix = self.stable.observe(self.partial_text, self.audio_time)
        if stable_prefix:
            return self._commit_prefix(stable_prefix)
        if self.partial_text != previous:
            return [self._event("partial", self.partial_text)]
        return []

    def commit_pending(self) -> list[TranscriptEvent]:
        if not self.partial_text:
            self.stable.reset()
            return []
        sentence = self.partial_text.rstrip()
        self.confirmed_text += sentence
        self.partial_text = ""
        self.stable.reset()
        return [
            self._event("sentence_final", sentence),
            self._event("partial", ""),
        ]

    def finish(self, text: str) -> list[TranscriptEvent]:
        events = self.apply_model_update(text, processed_samples=0)
        events.append(self._event("final", self.partial_text))
        return events

    def reset_segment(self) -> None:
        self.stable.reset()

    def _unconfirmed_tail(self, text: str) -> str:
        if not self.confirmed_text:
            return text
        if text.startswith(self.confirmed_text):
            return text[len(self.confirmed_text):]
        overlap = min(len(self.confirmed_text), len(text))
        while overlap:
            if self.confirmed_text.endswith(text[:overlap]):
                return text[overlap:]
            overlap -= 1
        raise ConfirmedPrefixConflict("model text conflicts with confirmed transcript prefix")

    def _commit_prefix(self, prefix: str) -> list[TranscriptEvent]:
        if not prefix or not self.partial_text.startswith(prefix):
            return []
        self.confirmed_text += prefix
        self.partial_text = self.partial_text[len(prefix):]
        self.stable.reset()
        return [
            self._event("sentence_final", prefix),
            self._event("partial", self.partial_text),
        ]

    def _event(self, event_type: str, text: str) -> TranscriptEvent:
        self._sequence += 1
        return TranscriptEvent(event_type, text, self._sequence)
```

Move the existing punctuation constants and helpers from `app/asr_api.py` into this module and expose `first_punctuation_candidate()`. Move `SilenceEndpointDetector`, `_pcm_s16le_rms()`, and duration calculation here unchanged, then rename private helpers without changing their behavior.

- [ ] **Step 4: Add edge-case tests before completing helpers**

Add tests for transient punctuation, punctuation removal, abbreviation/decimal/domain parsing, closing quotes, mixed terminators, overlap continuation, VAD one-shot behavior, and empty model revisions. Use explicit processed sample counts; no test may patch `time.monotonic()` or sleep.

- [ ] **Step 5: Run focused state tests**

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_asr_streaming.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/asr_streaming.py tests/test_asr_streaming.py
git commit -m "refactor: isolate deterministic asr transcript state"
```

---

### Task 3: Add Explicit ASR Model and Session Lifecycle

**Files:**
- Modify: `app/asr.py:70-320`
- Modify: `tests/test_asr_api.py`

- [ ] **Step 1: Write lifecycle tests**

Add three tests using concrete fake qwen-asr model and state classes:

- `test_qwen_vllm_warmup_loads_model_once` calls `warmup()` twice and asserts the fake `Qwen3ASRModel.LLM` constructor count is exactly one.
- `test_stateful_segment_reset_reinitializes_official_state` makes the first state return `"hello"`, resets, makes the second state return `" world"`, and asserts the wrapper returns cumulative `"hello world"`.
- `test_stateful_abort_releases_state` calls `abort()` and asserts a later `add_pcm_s16le()` raises `RuntimeError("streaming session is closed")`.

The fake model must record every `init_streaming_state()` result so the reset test also asserts that exactly two official state objects were created.

- [ ] **Step 2: Run lifecycle tests and verify failures**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py -k 'warmup or segment_reset or abort'
```

Expected: new lifecycle assertions fail.

- [ ] **Step 3: Extend interfaces**

Add to `ASRStreamingSession` and `ASRTranscriber`:

```python
class ASRStreamingSession:
    def add_pcm_s16le(self, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def finish(self) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def reset_segment(self) -> None:
        raise NotImplementedError

    def abort(self) -> None:
        raise NotImplementedError


class ASRTranscriber:
    def warmup(self) -> None:
        raise NotImplementedError
```

Implement `warmup()` by taking each existing model lock and calling `_load()`. Mock warmup is a no-op.

- [ ] **Step 4: Make stateful session reset and abort explicit**

Refactor `QwenVLLMStreamingSession` to keep:

```python
self._text_prefix = ""
self._closed = False
```

Every returned result uses:

```python
text=self._text_prefix + getattr(state, "text", "")
```

`reset_segment()` captures the current combined text into `_text_prefix` and asks the transcriber for a fresh official state. `abort()` clears the state reference and marks the session closed. `finish()` marks the session closed after returning the final combined result. Every mutating method checks `_closed` first.

Do not mutate `state.buffer` directly.

- [ ] **Step 5: Run ASR model-wrapper tests**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py -k 'qwen_vllm or segment_reset or abort or warmup'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/asr.py tests/test_asr_api.py
git commit -m "feat: define qwen asr session lifecycle"
```

---

### Task 4: Build the Single-Owner Inference Coordinator

**Files:**
- Create: `app/asr_inference.py`
- Create: `tests/test_asr_inference.py`

- [ ] **Step 1: Write coordinator ownership and ordering tests**

Use a fake transcriber that records `threading.get_ident()` in constructor, warmup, session initialization, add, finish, abort, and file transcription. Assert every recorded ID is identical and differs from the test thread.

Also add these concrete cases:

- `test_same_session_calls_complete_in_submission_order`: submit two chunks and assert the fake session records `[b"first", b"second"]`.
- `test_queue_full_raises_asr_queue_full`: block the worker, fill the configured one-slot queue, and assert the next submission raises `ASRQueueFull` without calling the model.
- `test_expired_queued_job_never_calls_model`: hold the worker past the queued job deadline and assert the fake call counter stays unchanged.
- `test_running_timeout_poisons_and_removes_session`: block an active call past its execution timeout, release it, and assert the next call raises `ASRSessionPoisoned` and the fake session's `abort()` count is one.
- `test_worker_continues_after_one_job_raises`: make one fake call raise, then assert the following independent call succeeds.
- `test_stop_rejects_new_jobs_and_joins_worker`: stop the coordinator, assert the worker is no longer alive, and assert new admission raises `ASRNotReady`.
- `test_file_job_is_rejected_while_stream_is_active`: create one stream and assert `transcribe_file()` raises `ASRFileTranscriptionDisabled` or the shared-instance admission error selected by configuration.
- `test_stream_is_rejected_while_file_job_is_running`: enable shared file transcription, block a file job in the worker, and assert `create_stream()` raises `ASRBatchConflict`.

Use `threading.Event` barriers instead of arbitrary sleeps wherever ordering matters.

- [ ] **Step 2: Run tests and verify import failure**

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_asr_inference.py
```

Expected: `ModuleNotFoundError: app.asr_inference`.

- [ ] **Step 3: Implement coordinator types**

Create `app/asr_inference.py` with these public contracts:

```python
class ASRCoordinatorError(RuntimeError):
    pass

class ASRNotReady(ASRCoordinatorError):
    pass

class ASRQueueFull(ASRCoordinatorError):
    pass

class ASRQueueTimeout(ASRCoordinatorError):
    pass

class ASRInferenceTimeout(ASRCoordinatorError):
    pass

class ASRSessionLimit(ASRCoordinatorError):
    pass

class ASRSessionPoisoned(ASRCoordinatorError):
    pass

class ASRFileTranscriptionDisabled(ASRCoordinatorError):
    pass

class ASRBatchConflict(ASRCoordinatorError):
    pass


@dataclass(frozen=True)
class CoordinatorSnapshot:
    ready: bool
    accepting: bool
    active_streams: int
    queue_depth: int
    queued_audio_seconds: float
    load_error: str | None
```

The completed `ASRInferenceCoordinator` exposes these exact methods:

- `__init__(settings: Settings, transcriber_factory: Callable[[], ASRTranscriber])`
- `async start() -> None`
- `async stop() -> None`
- `async create_stream(language: str | None) -> str`
- `async add_audio(session_id: str, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult`
- `async finish_stream(session_id: str) -> StreamingTranscriptionResult`
- `async reset_segment(session_id: str) -> None`
- `async abort_stream(session_id: str) -> None`
- `async transcribe_file(audio_path: str, language: str | None) -> TranscriptionResult`
- `snapshot() -> CoordinatorSnapshot`

Internally use a bounded `queue.PriorityQueue`, a monotonically increasing job sequence, `concurrent.futures.Future`, one `threading.Thread`, and a session dictionary owned only by the worker thread. Use numeric priority `0` for lifecycle/stream jobs, `10` for file jobs, and `100` for shutdown.

- [ ] **Step 4: Implement deadline-safe submission**

Before queue insertion, reserve queued audio seconds under a lock. Reject insertion when queue slots or global audio seconds are exhausted. Store an absolute deadline on each job. The worker checks cancellation and deadline before invoking any model method, then releases queue/audio capacity in `finally`.

Await futures using `asyncio.wrap_future()` and `asyncio.timeout()`. Shield a running future from cancellation. On an execution timeout, atomically mark the session poisoned; after the current call returns, the worker aborts and removes it before executing another job for that session.

- [ ] **Step 5: Add readiness and sanitized load errors**

The worker constructs the transcriber and calls `warmup()`. Store only `f"{type(exc).__name__}: {exc}"` after passing it through a helper that removes newlines and truncates to 300 characters. Never include API keys, audio bytes, or transcript text.

- [ ] **Step 6: Run coordinator tests**

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_asr_inference.py
```

Expected: all tests pass without leaked worker threads.

- [ ] **Step 7: Commit**

```bash
git add app/asr_inference.py tests/test_asr_inference.py
git commit -m "feat: serialize asr inference behind bounded coordinator"
```

---

### Task 5: Integrate Lifespan, Readiness, and File Admission

**Files:**
- Modify: `app/asr_api.py`
- Modify: `app/schemas.py`
- Modify: `tests/test_asr_api.py`

- [ ] **Step 1: Write readiness and responsiveness tests**

Add fake coordinators and dependency overrides, then implement these tests:

- `test_ready_returns_503_before_model_warmup` asserts status 503 and `status="not_ready"`.
- `test_ready_returns_snapshot_after_warmup` asserts status 200 plus active-stream, queue-depth, and queued-audio fields from a known fake snapshot.
- `test_file_transcribe_returns_503_when_disabled` asserts `detail.code == "file_transcription_disabled"` and confirms the fake model was not called.
- `test_file_transcribe_uses_async_coordinator` returns a known `TranscriptionResult` from the fake and asserts the temporary upload path was removed afterward.
- `test_health_remains_responsive_during_fake_slow_inference` blocks inference with a `threading.Event`, performs `/health` concurrently, and asserts health returns before the event is released.

The responsiveness test starts a fake inference blocked on an event, calls `/health` from another client/thread, and requires the health response before releasing inference.

- [ ] **Step 2: Run selected tests and verify failures**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py -k 'ready or responsive or file_transcribe'
```

Expected: new tests fail because coordinator integration is absent.

- [ ] **Step 3: Add lifespan and dependencies**

Create the transcriber only inside a factory passed to the coordinator. Add FastAPI lifespan:

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asr_coordinator.start()
    try:
        yield
    finally:
        await asr_coordinator.stop()
```

`start()` always starts the owner thread. When `asr_eager_load` is true, the worker calls `warmup()` before reporting ready. When it is false, the first admitted inference job performs warmup on the same owner thread; `/ready` remains 503 until that warmup succeeds.

Expose `get_asr_coordinator()` for dependency overrides. Tests must use `with TestClient(app) as client:` when they require lifespan.

- [ ] **Step 4: Add readiness response**

Add `ASRReadyResponse` to `app/schemas.py` with status, model, backend, active streams, queue depth, and queued audio seconds. `/ready` returns 503 when the coordinator is not ready or not accepting, otherwise 200.

Expose `file_transcribe_enabled` and `protocol_version` in `/v1/transcribe/stream-info` so clients and smoke scripts can discover the effective contract.

- [ ] **Step 5: Route file transcription through coordinator**

Replace the direct synchronous call with:

```python
result = await current_coordinator.transcribe_file(temp_path, language)
```

Map disabled, full, limit, and timeout errors to HTTP 503 with stable `detail.code` values. Always remove the uploaded temp file in `finally`.

- [ ] **Step 6: Run API and full mock suites**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q
```

Expected: all tests pass. Do not accept order-dependent environment failures; update ASR test fixtures so each ASR module import has explicit settings and cache cleanup.

- [ ] **Step 7: Commit**

```bash
git add app/asr_api.py app/schemas.py tests/test_asr_api.py
git commit -m "feat: add asr readiness and async inference admission"
```

---

### Task 6: Replace WebSocket Logic with a Versioned Session Controller

**Files:**
- Modify: `app/asr_api.py`
- Modify: `tests/test_asr_api.py`
- Modify: `app/asr_streaming.py`
- Modify: `tests/test_asr_streaming.py`

- [ ] **Step 1: Write strict protocol tests**

Add table-driven WebSocket tests for:

```text
non-JSON start                   -> error invalid_start, close 1003
sample_rate="bad"               -> error invalid_start, close 1003
sample_rate=8000                -> error unsupported_sample_rate, close 1003
language={}                     -> error invalid_language, close 1003
malformed JSON after ready      -> error invalid_message, connection remains usable or closes as documented
empty binary frame              -> error invalid_audio_frame
odd-length binary frame         -> error invalid_audio_frame
frame over ASR_MAX_FRAME_BYTES  -> error frame_too_large, close 1009
queue overload                  -> error server_busy, close 1013
session timeout                 -> error session_timeout, close 1008
confirmed-prefix conflict       -> error transcript_conflict, close 1011
```

Assert every version 2 server event has a strictly increasing `sequence`.

- [ ] **Step 2: Write protocol reconstruction tests**

Use fake cumulative model updates that add, revise, clear, punctuate, VAD-commit, segment-reset, and finish. Reconstruct client display text from events and assert it never duplicates confirmed content. Include a commit whose remaining partial is empty.

- [ ] **Step 3: Run new tests and verify failures**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py -k 'invalid or protocol or sequence or conflict or server_busy'
```

Expected: failures expose direct `int()` parsing, unhandled JSON, missing sequence, and direct model calls.

- [ ] **Step 4: Add structured start parsing**

Define a private Pydantic model in `app/asr_api.py`:

```python
class ASRStreamStart(BaseModel):
    type: Literal["start"]
    api_key: str
    language: str | None = Field(default=None, max_length=32)
    sample_rate: Literal[16000] = 16000
    format: Literal["pcm_s16le"] = "pcm_s16le"

    model_config = ConfigDict(extra="ignore")
```

Parse inside `asyncio.timeout(asr_start_timeout_seconds)` and convert Pydantic errors into stable protocol errors.

- [ ] **Step 5: Add `StreamingSessionController`**

Keep it in `app/asr_api.py` for this iteration. It owns coordinator session ID, `StreamingTranscriptState`, `SilenceEndpointDetector`, connection timestamps, and event sending. Its public methods are:

```python
async def start(self) -> None
async def add_audio(self, pcm_bytes: bytes) -> None
async def reset_segment(self) -> None
async def finish(self) -> None
async def abort(self) -> None
```

`add_audio()` validates bytes, awaits coordinator inference, passes `len(pcm_bytes) // 2` as processed samples to transcript state, then applies VAD. `finish()` awaits the official model finish before emitting final. `abort()` runs from a `finally` block and is idempotent.

Track accepted audio samples per connection. Reject the next frame with `audio_limit_exceeded` when it would cross `ASR_MAX_AUDIO_SECONDS`, and reject with `realtime_lag_exceeded` when coordinator queue wait or unprocessed connection audio exceeds `ASR_MAX_CONNECTION_LAG_SECONDS`.

- [ ] **Step 6: Map coordinator failures to protocol errors**

Add one error-sending helper that emits:

```json
{"type":"error","code":"server_busy","message":"ASR is at capacity","sequence":7}
```

Do not expose exception tracebacks or transcript text. Use close 1003 for client protocol errors, 1008 for policy/session timeout, 1009 for oversized frames, 1011 for model/state errors, and 1013 for overload.

- [ ] **Step 7: Enforce idle and maximum session duration**

Wrap each `websocket.receive()` with the smaller remaining duration of idle timeout and session timeout. On timeout, send the corresponding stable error and close. Always abort the coordinator session in `finally` unless finish already removed it.

- [ ] **Step 8: Run protocol and ASR tests**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_streaming.py tests/test_asr_inference.py tests/test_asr_api.py
```

Expected: all tests pass and no TestClient WebSocket hangs.

- [ ] **Step 9: Commit**

```bash
git add app/asr_api.py app/asr_streaming.py tests/test_asr_api.py tests/test_asr_streaming.py
git commit -m "feat: enforce versioned stateful asr protocol"
```

---

### Task 7: Fix Chunked Fallback Repeated-Text Loss

**Files:**
- Modify: `app/asr_api.py`
- Modify: `app/asr_streaming.py`
- Modify: `tests/test_asr_api.py`

- [ ] **Step 1: Add the known failing regression**

Mock two independent chunk transcriptions as `"你好"` and `"你好"`. Send enough audio for two chunk boundaries and assert the final display text is `"你好你好"`, not `"你好"`.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py -k repeated_independent_chunks
```

Expected: current prefix-based merging loses the second occurrence.

- [ ] **Step 3: Separate independent and cumulative APIs**

Provide an explicit state method:

```python
def append_independent_segment(self, text: str) -> list[TranscriptEvent]:
    if not text:
        return []
    self.partial_text += text
    return [self._event("partial", self.partial_text)]
```

Chunked mode uses this method. Stateful mode exclusively uses `apply_model_update()`.

- [ ] **Step 4: Run chunked and full ASR tests**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_api.py -k chunk
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q tests/test_asr_streaming.py tests/test_asr_api.py
```

Expected: repeated text is preserved and prior chunked tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/asr_api.py app/asr_streaming.py tests/test_asr_api.py
git commit -m "fix: preserve repeated text across asr chunks"
```

---

### Task 8: Update Client, Deployment Bounds, Smoke Checks, and Documentation

**Files:**
- Modify: `scripts/stream_asr_client.py`
- Modify: `tests/test_stream_asr_client.py`
- Modify: `scripts/smoke_asr.sh`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `cloud/README-A10.md`

- [ ] **Step 1: Add client display and sequence tests**

Test that `sentence_final` followed by `partial: ""` does not duplicate display text, sequence gaps are reported, and server error codes are printed without treating them as transcript text.

- [ ] **Step 2: Implement client protocol v2 handling**

Track `last_sequence`. Accept additive fields. Print a warning for a gap or non-increasing sequence, but continue displaying ordered WebSocket messages. Continue defining display as confirmed events plus the latest partial/final tail.

- [ ] **Step 3: Update smoke checks**

Have `scripts/smoke_asr.sh` call both `/health` and `/ready`. Add optional `EXPECT_ASR_PROTOCOL_VERSION=2` validation from stream info. A failed readiness check must make the script exit nonzero.

When `AUDIO_FILE` is set but stream info reports `file_transcribe_enabled=false`, exit with a clear nonzero message directing the operator to the dedicated batch instance. Do not treat the expected HTTP 503 from the live streaming service as a successful transcription smoke test.

- [ ] **Step 4: Bound the ASR WebSocket server in Compose**

Make the ASR command explicit:

```yaml
command:
  - python
  - -m
  - uvicorn
  - app.asr_api:app
  - --host
  - 0.0.0.0
  - --port
  - "8000"
  - --workers
  - "1"
  - --ws-max-size
  - "32000"
  - --ws-max-queue
  - "4"
```

Change only the ASR healthcheck URL to `/ready`. Keep its long start period because eager model loading is expected.

- [ ] **Step 5: Document the operational contract**

Document all new env variables, protocol version 2 events, close/error codes, one-process-per-GPU rule, audio-time stable punctuation, file transcription disabled by default, separate batch recommendation, readiness semantics, and A10 calibration procedure.

- [ ] **Step 6: Add privacy-safe operational telemetry**

Use module-level `logging.getLogger(__name__)` log events with fixed fields for session start/end, queue rejection, queue wait milliseconds, inference milliseconds, timeout category, confirmed-prefix conflict, and active stream count. Identify sessions with a random short session ID. Do not log API keys, audio bytes, file paths, language content, or transcript text. Add `caplog` tests that assert timing/error fields are present and a distinctive fake transcript/API key are absent.

- [ ] **Step 7: Run script and client tests**

```bash
bash -n scripts/smoke_asr.sh
PYTHONPATH=. .venv/bin/pytest -q tests/test_stream_asr_client.py
```

Expected: shell syntax and client tests pass.

- [ ] **Step 8: Run Compose validation when Docker is available**

```bash
docker compose config >/tmp/asr-compose-config.txt
```

Expected: exit 0 and rendered ASR command contains `--workers 1`, `--ws-max-size 32000`, and `/ready`.

- [ ] **Step 9: Commit**

```bash
git add scripts/stream_asr_client.py tests/test_stream_asr_client.py scripts/smoke_asr.sh docker-compose.yml README.md docs/API.md cloud/README-A10.md
git commit -m "docs: publish stateful asr protocol and limits"
```

---

### Task 9: Automated Regression and Static Verification

**Files:**
- Modify only if a failing check exposes an in-scope defect.

- [ ] **Step 1: Run all ASR tests**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q \
  tests/test_asr_config.py \
  tests/test_asr_streaming.py \
  tests/test_asr_inference.py \
  tests/test_asr_api.py \
  tests/test_stream_asr_client.py
```

Expected: all pass.

- [ ] **Step 2: Run the complete repository test suite**

```bash
PYTHONPATH=. API_KEY=test-key MODEL_BACKEND=mock .venv/bin/pytest -q
```

Expected: all pass. Record warnings separately; do not hide failures with `-x`, reruns, or ordering.

- [ ] **Step 3: Run syntax checks**

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q app scripts tests
bash -n scripts/bootstrap_ubuntu_gpu.sh scripts/deploy_remote.sh scripts/smoke_asr.sh scripts/smoke_test.sh scripts/update_all_services.sh scripts/update_service.sh
```

Expected: exit 0.

- [ ] **Step 4: Check diff hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Untracked user recordings and cost report remain untouched.

- [ ] **Step 5: Commit any narrowly required regression fix**

Use one commit per root cause. Do not combine unrelated cleanup.

---

### Task 10: Independent Test-Agent Acceptance

**Owner:** Dedicated test agent, not the implementation agent.

- [ ] **Step 1: Review without relying on implementation-agent claims**

Read the design, plan, commit diff, and public ASR protocol. Trace each design invariant to code and tests.

- [ ] **Step 2: Run the exact automated commands from Task 9**

Expected: clean pass. Report full failing command and root cause for any failure.

- [ ] **Step 3: Add adversarial tests where coverage is insufficient**

At minimum inspect and, when absent, test:

- timeout before execution versus timeout during execution;
- disconnect during a blocked inference call;
- queue accounting after exceptions;
- segment reset after confirmed and partial text;
- empty partial after every commit path;
- sequence behavior for error events;
- one slow stream alongside health/readiness requests;
- two sessions interleaving without state leakage.

- [ ] **Step 4: Verify no out-of-scope regressions**

Run translation and TTS mock tests even though those files should not change.

- [ ] **Step 5: Produce an acceptance report**

Report findings first by severity with file/line references. State one of:

```text
ACCEPTED: no known in-scope defect remains in available tests.
REJECTED: defects remain; list exact reproduction and required correction.
BLOCKED: name the unavailable external dependency and tests that remain unexecuted.
```

The test agent does not modify implementation unless the primary agent explicitly assigns a follow-up fix.

---

### Task 11: Primary-Agent Product Regression

**Owner:** Primary agent after implementation and independent test acceptance.

- [ ] **Step 1: Audit implementation and test-agent findings**

Confirm every rejected finding was fixed and independently rerun. Review final diff for scope, public API, error semantics, cleanup paths, and documentation consistency.

- [ ] **Step 2: Run product-level mock flow**

Start the ASR app in mock mode on a free port, then verify:

```text
GET /health
GET /ready
GET /v1/transcribe/stream-info
POST /v1/transcribe when file mode is disabled
WebSocket start -> ready -> audio -> partial -> sentence_final -> empty partial -> end -> final
invalid start and overload close-code behavior
```

- [ ] **Step 3: Run real Qwen/vLLM A10 acceptance when infrastructure exists**

Use both repository recordings in real-time mode:

```bash
API_KEY="$API_KEY" python3 scripts/stream_asr_client.py '录音3-小红.wav' \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language zh --print-mode events --show-stream-info --realtime

API_KEY="$API_KEY" python3 scripts/stream_asr_client.py '日语-单人16khz.wav' \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language ja --print-mode events --show-stream-info --realtime
```

Save protocol events outside git and verify reconstruction, no confirmed-text revision, no duplicate confirmed prefix, and clean finalization.

- [ ] **Step 4: Run A10 capacity calibration**

Measure 1, 2, 4, and 8 real-time streams. Record first-partial latency, queue wait p50/p95, inference p50/p95, real-time factor, maximum lag, GPU memory, and errors. Keep `ASR_MAX_ACTIVE_STREAMS=2` unless measured results support increasing it.

- [ ] **Step 5: Verify deployment path**

Build and recreate only the ASR service, check `/ready`, run `scripts/smoke_asr.sh`, inspect logs for sanitized errors, and verify Docker uses one Uvicorn worker.

- [ ] **Step 6: Apply completion gate**

Do not report completion until:

- automated tests pass;
- independent test agent accepts;
- product-level mock flow passes;
- real-model checks pass, or the user explicitly accepts a clearly stated GPU/model infrastructure block;
- no known in-scope defect remains;
- documentation matches effective defaults and protocol.

---

## Execution Orchestration Required by the User

Execution does not begin until the user explicitly says `可以执行` or an equivalent unambiguous instruction.

1. The primary agent invokes `superpowers:using-git-worktrees` and creates an isolated feature worktree without disturbing the current dirty workspace.
2. The primary agent spawns exactly one dedicated implementation agent and gives it this committed plan. That agent invokes `superpowers:executing-plans`, implements Tasks 1-9, and reports commits and verification evidence.
3. The primary agent reviews the implementation diff before testing handoff.
4. The primary agent spawns a different dedicated test agent for Task 10. The test agent receives the design and plan but is told not to trust the implementation report.
5. Rejected findings return to the implementation agent as narrowly scoped follow-up work; the test agent reruns acceptance afterward.
6. Once independently accepted, the primary agent performs Task 11 as the product owner and integration reviewer.
7. The primary agent reports completion only after every applicable gate passes. If real GPU/model infrastructure is unavailable, it reports the exact blocked checks and does not claim full production acceptance.
