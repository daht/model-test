# MT Dual Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two independent MT model processes behind the existing Uvicorn listener in the single `hy-mt-api` container.

**Architecture:** Add `--workers 2` to the MT image command. Uvicorn owns the existing container port and distributes connections between two processes; each process keeps its own translator and inference lock. No proxy, second container, or runtime setting is introduced.

**Tech Stack:** Dockerfile, Uvicorn, pytest

---

### Task 1: Pin the MT Container to Two Workers

**Files:**
- Create: `tests/test_mt_deployment.py`
- Modify: `Dockerfile:24`

- [ ] **Step 1: Write the failing deployment test**

Create `tests/test_mt_deployment.py` with:

```python
import json
from pathlib import Path


def test_mt_image_starts_exactly_two_uvicorn_workers() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD "))
    command = json.loads(cmd_line.removeprefix("CMD "))

    assert command[:3] == ["python", "-m", "uvicorn"]
    assert command[3] == "app.main:app"
    assert command.count("--workers") == 1
    worker_index = command.index("--workers")
    assert command[worker_index + 1] == "2"
```

- [ ] **Step 2: Run the test and verify the missing worker flag fails**

Run:

```bash
PATH="/model/.venv/bin:${PATH}" pytest -q tests/test_mt_deployment.py
```

Expected: FAIL because the current MT `CMD` does not contain `--workers`.

- [ ] **Step 3: Add the minimal MT startup change**

Replace the final `Dockerfile` command with:

```dockerfile
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

- [ ] **Step 4: Run focused deployment tests**

Run:

```bash
PATH="/model/.venv/bin:${PATH}" pytest -q tests/test_mt_deployment.py tests/test_api.py tests/test_model.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the implementation**

```bash
git add Dockerfile tests/test_mt_deployment.py
git commit -m "perf(mt): run two uvicorn workers"
```

### Task 2: Verify the Repository Release Gate

**Files:**
- Verify only: `Dockerfile`, `tests/test_mt_deployment.py`

- [ ] **Step 1: Validate the Compose model**

Run:

```bash
docker compose config --quiet
```

Expected: exit status 0.

- [ ] **Step 2: Run the repository commit gate**

Run:

```bash
ASR_MODEL_NAME=Qwen3-ASR-1.7B \
ASR_MAX_UTTERANCE_SECONDS=30 \
PATH="/model/.venv/bin:${PATH}" \
scripts/verify_asr_release.sh commit
```

Expected: all repository tests and release checks pass, including the unchanged ASR one-worker requirement.

- [ ] **Step 3: Check the final diff and repository state**

Run:

```bash
git diff --check HEAD^ HEAD
git show --stat --oneline HEAD
git status --short
```

Expected: the implementation commit contains only `Dockerfile` and `tests/test_mt_deployment.py`; existing unrelated untracked user files remain untouched.
