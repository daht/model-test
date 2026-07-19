# SenseVoice 128-Session Admission Implementation Plan

**Goal:** Allow the SenseVoice A10 test deployment to admit up to 128 sessions
without changing batching, rolling cadence, or runtime behavior.

**Scope:** Change the shared `ASR_MAX_ACTIVE_STREAMS` validation ceiling from 64
to 128, update the SenseVoice A10 example to 128, and prove that 128 is accepted
while 129 is rejected. The ignored root `.env` is updated only after the
committed candidate is accepted and merged.

## Task 1: Configuration Boundary

**Files:**
- Modify: `tests/test_asr_config.py`
- Modify: `app/config.py`

1. Add a failing boundary test:

```python
def test_active_stream_limit_accepts_128_and_rejects_129():
    settings = Settings(_env_file=None, asr_backend="mock", asr_max_active_streams=128)
    assert settings.asr_max_active_streams == 128
    with pytest.raises(ValidationError, match="less than or equal to 128"):
        Settings(_env_file=None, asr_backend="mock", asr_max_active_streams=129)
```

2. Run the test and require the 128 case to fail against the current ceiling.
3. Change only this declaration:

```python
asr_max_active_streams: int = Field(default=2, gt=0, le=128)
```

4. Re-run the focused configuration suite.

## Task 2: SenseVoice Test Example

**Files:**
- Modify: `tests/test_asr_deployment_scripts.py`
- Modify: `cloud/A10.sensevoice.env.example`

1. Require both example values to equal 128:

```python
assert "ASR_MAX_ACTIVE_STREAMS=128" in example
assert "ASR_GATEWAY_MAX_ACTIVE_SESSIONS=128" in example
```

2. Observe the test fail against the current 64-session example.
3. Replace only those two example values and re-run the deployment-script suite.

## Task 3: Candidate Gate And Integration

1. Stage exactly the four implementation/test paths.
2. Run `scripts/verify_asr_release.sh commit` while staged.
3. Commit the candidate and run independent read-only acceptance.
4. After acceptance, run fresh product regression and merge to `main`.
5. Set the ignored root `.env` values below and validate `Settings()` without
   printing the API key:

```dotenv
ASR_MAX_ACTIVE_STREAMS=128
ASR_GATEWAY_MAX_ACTIVE_SESSIONS=128
```

6. The server sweep remains 80, 96, 112, then 128; no result is inferred merely
   from configuration acceptance.
