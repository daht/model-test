# SenseVoice Small Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One Development Agent owns every implementation task and review correction required by the ASR delivery contract.

**Goal:** Add a locally loaded FunASR SenseVoice Small rolling WebSocket backend with dynamic micro-batching, normalized rich-transcription metadata, deterministic cleanup, and A10 release/test configuration.

**Architecture:** `SenseVoiceAdapter` is an independent Gateway worker adapter backed by one `SenseVoiceEngine` and one CUDA owner lock. The adapter accumulates each unconfirmed utterance, re-decodes it for replaceable partials, caches the final decode, and returns optional metadata through the existing scheduler and protocol. Gateway-owned VAD, session accounting, queueing, terminal behavior, and rollback backends remain unchanged.

**Tech Stack:** Python 3.11, FastAPI WebSocket Gateway, NumPy, PyTorch CUDA, `funasr==1.3.14`, pytest, Docker Compose, existing ASR release gates.

---

## Execution Contract

- Objective: produce one commit-verified SenseVoice Small backend candidate suitable for A10 release and live evaluation.
- Non-goals: native streaming, shared faster-whisper abstractions, diarization, timestamps, online model download, model weights, or capacity claims.
- Primary risk: repeated rolling decode can amplify utterance length; preserve exact per-session PCM limits and expose actual engine batch/elapsed metrics.
- Protocol risk: metadata must follow the current segment without changing payloads for backends that do not provide metadata.
- Runtime risk: FunASR batch result count/order and local model compatibility must fail closed.
- Intended source paths: `app/asr_sensevoice.py`, `app/asr_gateway_backends.py`, `app/asr_gateway_scheduler.py`, `app/asr_gateway_protocol.py`, `app/asr_gateway.py`, `app/config.py`.
- Intended test paths: `tests/test_asr_sensevoice.py`, `tests/test_asr_gateway_protocol.py`, `tests/test_asr_gateway.py`, `tests/test_asr_config.py`, `tests/test_asr_deployment_scripts.py`.
- Intended delivery paths: `requirements-asr-sensevoice.txt`, `Dockerfile.asr`, `cloud/A10.sensevoice.env.example`, `cloud/README-A10.md`, `docs/asr-release-verification.md`, `scripts/verify_asr_release.sh`.
- Forbidden paths: `.env`, audio, model directories, manifests generated from the target, logs, evidence archives, HTML reports, caches, and `superpowers/`.
- Local tests: focused pytest after each task, then explicit staged `scripts/verify_asr_release.sh commit`.
- External gates: A10 image build, pinned runtime check, approved model manifest, real non-silent warmup, strict speech, concurrency sweep, bottleneck monitor, and rollback remain unexecuted locally.
- Prerequisites: `scripts/verify_asr_release.sh`, `docs/asr-release-verification.md`, and the committed design `c51f164` are present.
- Checkpoints: at 30 minutes without a supported FunASR boundary, stop and report evidence; at 60 minutes without a testable candidate, split metadata plumbing from runtime integration.

## File Responsibility Map

- `app/asr_sensevoice.py`: FunASR construction/calls, tag normalization, per-session rolling buffers, batching, cleanup, and adapter observability.
- `app/asr_gateway_backends.py`: backend-neutral optional result metadata contract only.
- `app/asr_gateway_scheduler.py`: immutable propagation of metadata from adapter result to Gateway result.
- `app/asr_gateway_protocol.py`: serialize optional metadata onto transcript events without adding absent keys.
- `app/asr_gateway.py`: select the adapter and retain/clear metadata at segment boundaries.
- `app/config.py`: validate only the two SenseVoice-specific settings and backend/stream pair.
- `requirements-asr-sensevoice.txt` and `Dockerfile.asr`: pin and install the runtime.
- `cloud/A10.sensevoice.env.example`: test-environment configuration without credentials.
- release docs and runner: accept and verify the new local-model backend.

### Task 1: Add The Optional Metadata Contract

**Files:**
- Modify: `app/asr_gateway_backends.py`
- Modify: `app/asr_gateway_scheduler.py`
- Modify: `app/asr_gateway_protocol.py`
- Test: `tests/test_asr_gateway_protocol.py`

- [ ] **Step 1: Write failing protocol tests for optional metadata**

Add tests proving metadata is present only when supplied:

```python
def test_optional_result_metadata_is_attached_without_changing_plain_events():
    protocol = ProtocolSession(sample_rate=16000, segment_local=True)
    plain = protocol.apply_result(
        ResultMode.REPLACEABLE_SEGMENT,
        text="plain",
        decoded_samples=2,
        segment_id=1,
    )
    rich = protocol.apply_result(
        ResultMode.REPLACEABLE_SEGMENT,
        text="rich",
        decoded_samples=2,
        segment_id=1,
        metadata={"language": "zh", "emotion": "neutral", "audio_event": "speech"},
    )
    committed = protocol.segment(metadata={"language": "zh"})
    final = protocol.final(metadata={"language": "zh"})

    assert all("metadata" not in event for event in plain)
    assert rich[-1]["metadata"]["emotion"] == "neutral"
    assert committed[0]["metadata"] == {"language": "zh"}
    assert final["metadata"] == {"language": "zh"}
```

- [ ] **Step 2: Run the test and verify the signatures fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_gateway_protocol.py::test_optional_result_metadata_is_attached_without_changing_plain_events -q
```

Expected: `FAIL` because `apply_result`, `segment`, and `final` do not accept `metadata`.

- [ ] **Step 3: Add metadata to backend-neutral results and protocol serialization**

Add `metadata: Mapping[str, Any] | None = None` to both `AdapterResult` and
`InferenceResult`, importing `Mapping` where necessary. Update the protocol
methods with this exact omission rule:

```python
def apply_result(..., metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    ...
    extra = {"metadata": dict(metadata)} if metadata is not None else {}
    return [self._serialize(event, **extra) for event in events]

def segment(self, *, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    self._require_open()
    extra = {"metadata": dict(metadata)} if metadata is not None else {}
    return [self._serialize(event, **extra) for event in self.state.commit_pending()]

def final(
    self,
    tail_text: str | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ...
    extra = {"metadata": dict(metadata)} if metadata is not None else {}
    event = self._serialize(self.state.final_event(), **extra)
```

Do not introduce a generic metadata framework or validation registry.

- [ ] **Step 4: Run protocol and scheduler/backends regression**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_gateway_protocol.py tests/test_asr_gateway_scheduler.py tests/test_asr_gateway_backends.py -q
```

Expected: all tests pass; existing events without metadata have no new key.

### Task 2: Add SenseVoice Configuration And Runtime Dependency

**Files:**
- Modify: `app/config.py`
- Create: `requirements-asr-sensevoice.txt`
- Modify: `Dockerfile.asr`
- Test: `tests/test_asr_config.py`
- Test: `tests/test_asr_deployment_scripts.py`

- [ ] **Step 1: Write failing configuration and image-contract tests**

Add these cases:

```python
def test_sensevoice_configuration_contract():
    settings = Settings(
        _env_file=None,
        asr_backend="sensevoice",
        asr_stream_mode="rolling",
        asr_model_name="SenseVoiceSmall",
        asr_model_id="/models/SenseVoiceSmall",
        asr_sensevoice_batch_size=8,
        asr_sensevoice_use_itn=True,
        api_key=TEST_ONLY_LONG_API_KEY,
    )
    assert settings.asr_sensevoice_batch_size == 8
    assert settings.asr_sensevoice_use_itn is True


def test_sensevoice_requires_rolling_streaming():
    with pytest.raises(ValidationError, match="rolling"):
        Settings(
            _env_file=None,
            asr_backend="sensevoice",
            asr_stream_mode="stateful",
            api_key=TEST_ONLY_LONG_API_KEY,
        )
```

In `tests/test_asr_deployment_scripts.py`, assert
`requirements-asr-sensevoice.txt` contains `funasr==1.3.14`, the Dockerfile
copies it, and pip installs it.

- [ ] **Step 2: Run the new tests and verify validation/file failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_config.py -k sensevoice -q
.venv/bin/python -m pytest tests/test_asr_deployment_scripts.py -k sensevoice -q
```

Expected: `FAIL` for unsupported backend/settings and missing requirements file.

- [ ] **Step 3: Implement minimal configuration**

Change only these declarations and validators:

```python
asr_backend: Literal[
    "qwen", "qwen_vllm", "faster_whisper", "sensevoice", "mock"
] = "qwen"
asr_sensevoice_batch_size: int = Field(default=8, gt=0, le=64)
asr_sensevoice_use_itn: bool = True
```

Include `sensevoice` in production API-key/manifest validation and require
rolling mode:

```python
if self.asr_backend == "sensevoice" and self.asr_stream_mode != "rolling":
    raise ValueError("asr_backend=sensevoice requires rolling streaming")
```

Create `requirements-asr-sensevoice.txt` containing only:

```text
funasr==1.3.14
```

Extend the existing Dockerfile `COPY` and pip install lists with
`requirements-asr-sensevoice.txt`; do not add a second image or download a
model during build.

- [ ] **Step 4: Run focused configuration and deployment tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_config.py tests/test_asr_deployment_scripts.py -q
```

Expected: all tests pass.

### Task 3: Implement SenseVoice Text Normalization And Engine Boundary

**Files:**
- Create: `app/asr_sensevoice.py`
- Create: `tests/test_asr_sensevoice.py`

- [ ] **Step 1: Write failing normalization and engine tests**

Use a fake `funasr.AutoModel` installed through `sys.modules`. Cover known tags,
unknown tags, batch order, ITN forwarding, and count mismatch:

```python
def test_normalize_sensevoice_output_extracts_tags_and_clean_text():
    decoded = normalize_sensevoice_output(
        "<|zh|><|NEUTRAL|><|Speech|>今天天气不错。"
    )
    assert decoded == SenseVoiceDecoded(
        "今天天气不错。",
        {"language": "zh", "emotion": "neutral", "audio_event": "speech"},
    )


def test_normalize_sensevoice_output_omits_unknown_tags():
    decoded = normalize_sensevoice_output("<|xx|><|EMO_UNKNOWN|>hello")
    assert decoded.text == "hello"
    assert decoded.metadata == {}


def test_engine_uses_local_batch_api_and_preserves_order(monkeypatch, tmp_path):
    model = install_fake_funasr(monkeypatch, [
        {"text": "<|zh|><|NEUTRAL|><|Speech|>甲"},
        {"text": "<|zh|><|HAPPY|><|Laughter|>乙"},
    ])
    engine = SenseVoiceEngine(str(tmp_path), device="cuda:0", use_itn=True)
    result = engine.transcribe_batch(
        [np.zeros(1600, dtype=np.float32), np.ones(800, dtype=np.float32)],
        language="zh",
    )
    assert [item.text for item in result] == ["甲", "乙"]
    assert model.generate_calls[0]["language"] == "zh"
    assert model.generate_calls[0]["use_itn"] is True
    assert model.generate_calls[0]["batch_size"] == 2
```

- [ ] **Step 2: Run the new test module and verify missing-module failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_sensevoice.py -q
```

Expected: collection fails because `app.asr_sensevoice` does not exist.

- [ ] **Step 3: Implement the minimal engine and parser**

Define:

```python
@dataclass(frozen=True)
class SenseVoiceDecoded:
    text: str
    metadata: dict[str, str]


LANGUAGE_TAGS = {"zh", "yue", "en", "ja", "ko"}
EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}
EVENT_TAGS = {
    "Speech", "BGM", "Applause", "Laughter", "Cry", "Cough", "Sneeze"
}
TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")


def normalize_sensevoice_output(raw: str) -> SenseVoiceDecoded:
    tags = TAG_PATTERN.findall(raw)
    metadata: dict[str, str] = {}
    for tag in tags:
        if tag in LANGUAGE_TAGS and "language" not in metadata:
            metadata["language"] = tag
        elif tag in EMOTION_TAGS and "emotion" not in metadata:
            metadata["emotion"] = tag.lower()
        elif tag in EVENT_TAGS and "audio_event" not in metadata:
            metadata["audio_event"] = tag.lower()
    return SenseVoiceDecoded(TAG_PATTERN.sub("", raw).strip(), metadata)
```

`SenseVoiceEngine.__init__` lazily imports `AutoModel` and constructs exactly
one local model:

```python
self._model = AutoModel(
    model=model_id,
    device=device,
    trust_remote_code=False,
    disable_update=True,
)
```

`transcribe_batch` calls:

```python
raw = self._model.generate(
    input=list(audio),
    cache={},
    language=language or "auto",
    use_itn=self._use_itn,
    batch_size=len(audio),
)
```

Require a list with one mapping containing a string `text` per input. Otherwise
raise a sanitized internal `RuntimeError` with no raw model output. `close`
drops the model reference and empties CUDA cache only if torch reports CUDA
available.

For warmup, require `${model_id}/example/en.mp3`, call the same model's
`generate` path with language `en`, and require non-empty cleaned text plus a
recognized language tag. This is the official model-bundled speech sample, not
a generated tone or repository-tracked audio.

- [ ] **Step 4: Run engine/parser tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_sensevoice.py -q
```

Expected: parser and engine-boundary tests pass without installing FunASR.

### Task 4: Implement Rolling Dynamic-Microbatch Adapter And Cleanup

**Files:**
- Modify: `app/asr_sensevoice.py`
- Modify: `tests/test_asr_sensevoice.py`

- [ ] **Step 1: Write failing adapter tests**

Add a recording engine and helpers modeled on `tests/test_asr_faster_whisper.py`.
Cover these exact behaviors:

```python
def test_partial_redecodes_full_accumulated_utterance_and_replaces_text():
    # First call receives 3 samples; second call receives the accumulated 5.
    # The second InferenceResult.text is only the second full snapshot.
    assert calls == [([3], "zh"), ([5], "zh")]
    assert [first[0].text, second[0].text] == ["text-3", "text-5"]


def test_cross_session_batch_preserves_identity_and_metadata():
    assert [item.session_id for item in results] == ["a", "b"]
    assert [item.metadata["language"] for item in results] == ["zh", "ja"]


def test_final_batch_is_cached_then_finish_clears_pcm_and_metadata():
    assert finish.text == final_result.text
    assert finish.metadata == final_result.metadata
    assert snapshot["session_audio_samples"] == 0


def test_abort_close_stale_identity_overflow_and_result_count_fail_closed():
    # Assert stale IDs and duplicate session jobs fail, exact utterance limits
    # reject before extending PCM, and abort/close leave zero session samples.
```

Also test manifest-before-engine warmup, unsupported task/timestamps/language,
one job per session per batch, explicit segment fallback final decode, engine
observer fields, cancellation safety, and session removal after finish.

- [ ] **Step 2: Run adapter tests and verify missing adapter failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_sensevoice.py -q
```

Expected: new adapter tests fail because `SenseVoiceAdapter` is absent.

- [ ] **Step 3: Implement one focused adapter without a shared base class**

Use this state and capability shape:

```python
@dataclass
class _SessionState:
    backend_session_id: str
    language: str | None
    pcm: bytearray
    cached_final: SenseVoiceDecoded | None = None


self.capabilities = BackendCapabilities(
    protocol_version=1,
    worker_id=worker_id,
    model_id=model_id,
    model_revision=model_revision,
    gpu_id=gpu_id,
    languages=("auto", "zh", "yue", "en", "ja", "ko"),
    tasks=("transcribe",),
    streaming_mode=StreamingMode.ROLLING,
    dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH,
    vad_mode=VadMode.GATEWAY,
    result_mode=ResultMode.REPLACEABLE_SEGMENT,
    preferred_chunk_samples=32_000,
    max_input_samples=max_segment_samples,
    max_segment_samples=max_segment_samples,
    max_batch_items=batch_size,
    max_batch_samples=max_segment_samples * batch_size,
    max_in_flight=1,
    session_capacity=session_capacity,
    retry_safe=False,
    warmed=False,
    backend_id="local",
)
```

`submit` validates every reservation before extending any buffer. It then
groups indices by session language, converts each complete accumulated PCM
buffer with `np.frombuffer(..., dtype="<i2").astype(np.float32) / 32768.0`,
performs one engine call per language group under `_engine_lock`, checks result
counts, restores original order, caches final items, and returns:

```python
InferenceResult.from_job(
    job,
    text=item.text,
    tail_text=item.text,
    segment_id=1,
    final=job.final,
    metadata=item.metadata or None,
)
```

`finish_segment` consumes a cached final or performs one final full-buffer
decode, returns `AdapterResult(..., metadata=item.metadata or None)`, and then
clears PCM/cache. `finish_session` additionally removes the session.
`abort_session` and `close` remove all state. Follow the existing adapter's
capacity event names and engine observer interface, changing only the component
to `sensevoice_adapter`.

- [ ] **Step 4: Run the complete SenseVoice adapter suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_sensevoice.py -q
```

Expected: all tests pass and no FunASR/model/GPU prerequisite is needed.

### Task 5: Wire Metadata Lifecycle And Runtime Selection Into The Gateway

**Files:**
- Modify: `app/asr_gateway.py`
- Modify: `tests/test_asr_gateway.py`
- Modify: `tests/test_asr_gateway_protocol.py`

- [ ] **Step 1: Write failing Gateway metadata and runtime-selection tests**

Add a `sensevoice` runtime selection assertion parallel to faster-whisper and a
Gateway scenario with two segments:

```python
def test_default_runtime_selects_sensevoice_and_preserves_rollback(monkeypatch):
    settings = Settings(
        _env_file=None,
        model_backend="mock",
        asr_backend="sensevoice",
        asr_stream_mode="rolling",
        asr_model_name="SenseVoiceSmall",
        asr_model_id="/models/SenseVoiceSmall",
        asr_sensevoice_batch_size=8,
        api_key="unit-test-only-not-a-production-secret-000000",
    )
    monkeypatch.setattr("app.asr_gateway.get_settings", lambda: settings)
    runtime = _default_runtime()
    assert isinstance(runtime.adapters["local"], SenseVoiceAdapter)
    assert runtime.adapters["local"].capabilities.max_batch_items == 8
```

The lifecycle test must prove a result metadata object reaches `partial`, the
same segment reaches `sentence_final` and `final`, a new segment replaces the
saved metadata, and a plain fake adapter still emits no metadata key.

- [ ] **Step 2: Run focused tests and verify selection/lifecycle failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_gateway.py -k 'sensevoice or metadata' -q
.venv/bin/python -m pytest tests/test_asr_gateway_protocol.py -q
```

Expected: `FAIL` because runtime selection and Gateway metadata state are absent.

- [ ] **Step 3: Implement the minimal Gateway lifecycle**

Add one context field:

```python
segment_metadata: dict[str, Any] | None = None
```

When an `InferenceResult` has metadata, copy it into this field and pass it to
`ProtocolSession.apply_result`. Extend `_apply_control_result` to accept
metadata and pass it to the protocol. At endpoint or explicit segment, prefer
the control result's metadata, attach the retained value to `segment()`, then
clear it after commit. At finish, attach the latest control/retained metadata
to `final()` and clear it after serialization. Never synthesize an empty
metadata object.

Add `_default_runtime` branch:

```python
if settings.asr_backend == "sensevoice":
    from app.asr_sensevoice import SenseVoiceAdapter, SenseVoiceEngine
    max_segment_samples = round(settings.asr_max_utterance_seconds * 16_000)
    adapter = SenseVoiceAdapter(
        lambda: SenseVoiceEngine(
            settings.asr_model_id,
            device=settings.asr_device,
            use_itn=settings.asr_sensevoice_use_itn,
        ),
        worker_id="local",
        model_id=settings.asr_model_id,
        model_revision=settings.asr_model_name,
        gpu_id=settings.asr_device,
        session_capacity=settings.asr_max_active_streams,
        batch_size=settings.asr_sensevoice_batch_size,
        max_segment_samples=max_segment_samples,
        model_manifest_path=settings.asr_model_manifest_path,
    )
    return GatewayRuntime(settings, {"local": adapter})
```

Do not change Qwen or faster-whisper construction.

- [ ] **Step 4: Run Gateway-focused regression**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_gateway_protocol.py tests/test_asr_gateway.py tests/test_asr_gateway_scheduler.py -q
```

Expected: all tests pass, including cleanup and exactly-one-terminal-final cases.

### Task 6: Add A10 Configuration, Release Contract, And Operator Runbook

**Files:**
- Create: `cloud/A10.sensevoice.env.example`
- Modify: `scripts/verify_asr_release.sh`
- Modify: `docs/asr-release-verification.md`
- Modify: `cloud/README-A10.md`
- Modify: `tests/test_asr_deployment_scripts.py`

- [ ] **Step 1: Write failing deployment contract tests**

Assert the new example includes:

```python
required = {
    "ASR_BACKEND=sensevoice",
    "ASR_STREAM_MODE=rolling",
    "ASR_MODEL_NAME=SenseVoiceSmall",
    "ASR_MODEL_ID=/models/SenseVoiceSmall",
    "ASR_MODEL_MANIFEST_PATH=/models/SenseVoiceSmall.manifest.json",
    "ASR_SENSEVOICE_BATCH_SIZE=8",
    "ASR_SENSEVOICE_USE_ITN=true",
    "ASR_MAX_UTTERANCE_SECONDS=15.0",
    "ASR_GATEWAY_DEFAULT_UPDATE_MS=2000",
    "ASR_GATEWAY_MAX_ACTIVE_SESSIONS=64",
}
```

Also assert `API_KEY=\n`, release stream contract
`"sensevoice": "rolling"`, exact SenseVoice settings, the pinned FunASR R06
check, model/manifest staging instructions, the `1/8/16/24/32/64` sweep, and
atomic rollback instructions.

- [ ] **Step 2: Run deployment tests and verify missing-contract failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_deployment_scripts.py -k sensevoice -q
```

Expected: `FAIL` for missing example, runner contract, and documentation.

- [ ] **Step 3: Implement the exact release and test-environment contract**

Create the example by retaining the common service/TTS/VAD fields from
`cloud/A10.faster-whisper.env.example`, replacing only backend/model settings
and setting test admission to 64. Keep batch size eight, 15-second utterances,
2-second rolling updates, 200 ms maximum batch collection delay, six seconds
per-session buffer, and sufficiently bounded queue limits for 64 admitted
streams. Leave `API_KEY` empty.

Extend release validation:

```python
stream_contracts = {
    "qwen_vllm": "stateful",
    "faster_whisper": "rolling",
    "sensevoice": "rolling",
}
if backend == "sensevoice":
    required.update({
        "ASR_SENSEVOICE_BATCH_SIZE": "8",
        "ASR_SENSEVOICE_USE_ITN": "true",
        "ASR_MAX_UTTERANCE_SECONDS": "15.0",
        "ASR_GATEWAY_DEFAULT_UPDATE_MS": "2000",
    })
```

Add R06:

```bash
docker run --rm --entrypoint python qwen-asr-api:latest -c \
  'import importlib.metadata as m; assert m.version("funasr") == "1.3.14"'
```

Document trusted/approved model staging and manifest creation for
`FunAudioLLM/SenseVoiceSmall`, runtime local-only loading, release environment
paths, strict multilingual speech checks, the concurrency sweep, monitor use,
quality comparison, and rollback to the prior matching backend/image/model/
manifest/config. Do not include an API key or tell operators to place one in
argv.

- [ ] **Step 4: Run deployment tests and shell syntax**

Run:

```bash
.venv/bin/python -m pytest tests/test_asr_deployment_scripts.py -q
bash -n scripts/verify_asr_release.sh
```

Expected: all tests and shell syntax pass.

### Task 7: Run Full Candidate Gate And Commit

**Files:**
- All intended implementation, test, dependency, configuration, runner, and documentation paths listed in the execution contract.

- [ ] **Step 1: Run focused combined regression**

Run:

```bash
MODEL_BACKEND=mock ASR_BACKEND=mock ASR_STREAM_MODE=chunked \
ASR_REQUIRE_MODEL_MANIFEST=false ASR_MODEL_MANIFEST_PATH= \
TTS_BACKEND=mock API_KEY=test-key \
.venv/bin/python -m pytest \
  tests/test_asr_sensevoice.py \
  tests/test_asr_gateway_protocol.py \
  tests/test_asr_gateway.py \
  tests/test_asr_gateway_scheduler.py \
  tests/test_asr_config.py \
  tests/test_asr_deployment_scripts.py -q
```

Expected: all focused tests pass once.

- [ ] **Step 2: Inspect the exact intended delta**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: only intended tracked paths are modified/new; pre-existing untracked
audio, logs, reports, and load script remain untouched.

- [ ] **Step 3: Stage only intended files**

Run:

```bash
git add -- \
  app/asr_sensevoice.py \
  app/asr_gateway_backends.py \
  app/asr_gateway_scheduler.py \
  app/asr_gateway_protocol.py \
  app/asr_gateway.py \
  app/config.py \
  tests/test_asr_sensevoice.py \
  tests/test_asr_gateway_protocol.py \
  tests/test_asr_gateway.py \
  tests/test_asr_config.py \
  tests/test_asr_deployment_scripts.py \
  requirements-asr-sensevoice.txt \
  Dockerfile.asr \
  cloud/A10.sensevoice.env.example \
  cloud/README-A10.md \
  docs/asr-release-verification.md \
  scripts/verify_asr_release.sh
```

Expected: `git diff --cached --name-only` lists exactly these paths.

- [ ] **Step 4: Run the authoritative staged commit gate**

Run:

```bash
scripts/verify_asr_release.sh commit
```

Expected: C01-C06 pass. Stop at the first failure; do not weaken a threshold or
raise a size limit after observing failure.

- [ ] **Step 5: Commit the candidate and report its SHA**

Run:

```bash
git commit -m "feat(asr): add SenseVoice Small backend"
git rev-parse HEAD
git status --short
```

Expected: one implementation commit exists; only the same pre-existing
untracked files remain.

### Task 8: Independent Acceptance And Product Regression

**Files:**
- Read-only verification of the committed SHA.

- [ ] **Step 1: Prepare a clean detached acceptance worktree**

The primary agent creates a detached worktree outside the dirty root checkout
at the exact candidate SHA. The Test Agent receives only the SHA, worktree,
contract, relevant paths, and required invariants.

- [ ] **Step 2: Run independent read-only acceptance**

The Test Agent runs the focused tests and `scripts/verify_asr_release.sh commit`,
checks metadata isolation, rolling full-buffer behavior, result-count failure,
cleanup, terminal protocol, secrets, forbidden artifacts, residue, and Git
status, then returns exactly `ACCEPTED` or `REJECTED` with reproducible evidence.

- [ ] **Step 3: Run fresh primary-agent product regression after acceptance**

Run the focused combined regression and commit gate again against the accepted
SHA. Confirm protocol terminal behavior, zero retained session samples after
finish/abort, absent metadata for non-SenseVoice results, and no secret or
forbidden artifact in the commit.

- [ ] **Step 4: Report integration and external gate state**

Report candidate SHA, focused and commit-gate evidence, acceptance verdict,
product regression, preserved user files, and that release/live A10 gates are
unexecuted until the model, manifest, Docker/GPU server, deployed URL, runtime
credential, external speech, and thresholds are available.
