import os
import shutil
import subprocess
from pathlib import Path


RUNNER = Path("scripts/deploy_asr_cloud.sh")


def _run(*args, env=None, cwd=None):
    return subprocess.run(
        ["bash", str(RUNNER.resolve()), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_and_dry_run_are_deterministic_without_cloud_prerequisites():
    help_result = _run("--help", env={"PATH": os.environ["PATH"]})
    assert help_result.returncode == 0
    assert "Usage:" in help_result.stdout
    assert "--dry-run" in help_result.stdout
    assert "--skip-live" in help_result.stdout
    assert "ASR_LIVE_API_KEY" in help_result.stdout

    dry_result = _run("--dry-run", env={"PATH": os.environ["PATH"]})
    assert dry_result.returncode == 0
    assert "DRY RUN: no prerequisites checked" in dry_result.stdout
    ordered = [
        "validate clean committed checkout and external release/live inputs",
        "prepare repository .venv with pinned requirements-dev.txt when needed",
        "snapshot current deployment image and back up .env plus approved manifest",
        "verify and receipt the running rollback image/config/model/manifest baseline",
        "stop the existing ASR model owner and begin the maintenance window",
        "run scripts/verify_asr_release.sh release",
        "deploy the exact release-verified image without rebuilding",
        "verify local readiness and WebSocket smoke",
        "run evidence-bound deployed-live gates without another model owner",
    ]
    positions = [dry_result.stdout.index(item) for item in ordered]
    assert positions == sorted(positions)


def _write_executable(path: Path, content: str):
    path.write_text(content)
    path.chmod(0o755)


def _make_harness(tmp_path: Path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    state = tmp_path / "state"
    secure = tmp_path / "secure"
    model = repo / "models" / "Qwen3-ASR-1.7B"
    for path in (scripts, fake_bin, state, secure, model):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copy2(RUNNER, scripts / RUNNER.name)
    shutil.copy2(Path("scripts/asr_deploy_receipt.py"), scripts / "asr_deploy_receipt.py")
    (repo / "requirements-dev.txt").write_text("pytest==9.1.1\n")
    (repo / ".env").write_text("API_KEY=stored-outside-command-line\n")
    (repo / "models" / "Qwen3-ASR-1.7B.manifest.json").write_text(
        '{"revision":"approved"}\n'
    )
    zh_audio = secure / "zh.flac"
    ja_audio = secure / "ja.flac"
    zh_audio.write_bytes(b"zh-speech")
    ja_audio.write_bytes(b"ja-speech")
    (state / "latest").write_text("sha256:old\n")
    (state / "deployed").write_text("sha256:old\n")
    (state / "events").write_text("")

    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -eu
printf 'git:%s\\n' "$*" >>"$FAKE_STATE/events"
case "${1:-}" in
  status)
    if [[ "${FAKE_DIRTY:-0}" == 1 ]]; then
      echo ' M app/asr.py'
    fi
    ;;
  rev-parse)
    case "${2:-}" in
      --show-toplevel) printf '%s\\n' "$FAKE_REPO" ;;
      --is-inside-work-tree) echo true ;;
      *) echo 0123456789abcdef0123456789abcdef01234567 ;;
    esac
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "python3",
        """#!/usr/bin/env bash
set -eu
printf 'python:%s\\n' "$*" >>"$FAKE_STATE/events"
if [[ "${1:-}" == -m && "${2:-}" == venv ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
  exit 0
fi
if [[ "${1:-}" == -m && "${2:-}" == pip ]]; then
  touch "$FAKE_STATE/pytest-installed"
  exit 0
fi
if [[ "${1:-}" == -c && "$2" == *pytest* ]]; then
  [[ -f "$FAKE_STATE/pytest-installed" ]] || exit 1
  echo 9.1.1
  exit 0
fi
if [[ "${1:-}" == -m && "${2:-}" == app.asr_artifacts ]]; then
  printf 'asset-verify\\n' >>"$FAKE_STATE/events"
  exit "${FAKE_ASSET_VERIFY_STATUS:-0}"
fi
if [[ "$*" == *scripts/asr_deploy_receipt.py* ]]; then
  if [[ "$*" == *validate-running-baseline* ]]; then
    printf 'receipt:validate-baseline\\n' >>"$FAKE_STATE/events"
    exit "${FAKE_BASELINE_MISMATCH:-0}"
  fi
  if [[ "$*" == *create-receipt* ]]; then
    printf 'receipt:create\\n' >>"$FAKE_STATE/events"
    previous=""
    for argument in "$@"; do
      if [[ "$previous" == --output ]]; then printf '{}\\n' >"$argument"; break; fi
      previous="$argument"
    done
    exit 0
  fi
  if [[ "$*" == *validate-receipt* ]]; then
    printf 'receipt:validate\\n' >>"$FAKE_STATE/events"
    exit "${FAKE_RECEIPT_STATUS:-0}"
  fi
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
printf 'docker:%s\\n' "$*" >>"$FAKE_STATE/events"
if [[ "${1:-}" == compose && "$*" == *' ps -q qwen-asr-api' ]]; then
  deployed="$(cat "$FAKE_STATE/deployed")"
  if [[ "$deployed" == sha256:old ]]; then echo oldcid; fi
  if [[ "$deployed" == sha256:new ]]; then echo newcid; fi
  exit 0
fi
if [[ "${1:-}" == inspect && "$*" == *'--format'* && "$*" == *oldcid ]]; then echo sha256:old; exit 0; fi
if [[ "${1:-}" == inspect && "$*" == *'--format'* && "$*" == *newcid ]]; then cat "$FAKE_STATE/deployed"; exit 0; fi
if [[ "${1:-}" == inspect && "$*" == *oldcid ]]; then
  mismatch_value=qwen_vllm
  if [[ "${FAKE_BASELINE_MISMATCH:-0}" == 1 ]]; then mismatch_value=mock; fi
  cat <<JSON
[{"Image":"sha256:old","Config":{"Env":["API_KEY=stored-outside-command-line-1234567890","ASR_BACKEND=$mismatch_value","ASR_STREAM_MODE=stateful","ASR_REQUIRE_MODEL_MANIFEST=true","ASR_EAGER_LOAD=true","ASR_FILE_TRANSCRIBE_ENABLED=false","ASR_MODEL_ID=/models/Qwen3-ASR-1.7B","ASR_MODEL_MANIFEST_PATH=/models/Qwen3-ASR-1.7B.manifest.json"],"Cmd":["python","-m","uvicorn","app.asr_api:app","--workers","1"]},"State":{"Running":true,"StartedAt":"2099-01-01T00:00:00Z"},"Mounts":[{"Destination":"/models","Source":"$FAKE_REPO/models","RW":false}]}]
JSON
  exit 0
fi
if [[ "${1:-}" == image && "${2:-}" == inspect ]]; then cat "$FAKE_STATE/latest"; exit 0; fi
if [[ "${1:-}" == tag ]]; then
  source_image="$2"
  target_image="$3"
  if [[ "$source_image" == sha256:old || "$source_image" == *:rollback-* ]]; then
    resolved=sha256:old
  else
    resolved=sha256:new
  fi
  if [[ "$target_image" == qwen-asr-api:latest ]]; then printf '%s\\n' "$resolved" >"$FAKE_STATE/latest"; fi
  exit 0
fi
if [[ "${1:-}" == compose && "$*" == *' config --format json'* ]]; then
  cat <<JSON
{"services":{"qwen-asr-api":{"environment":{"API_KEY":"stored-outside-command-line-1234567890","ASR_BACKEND":"qwen_vllm","ASR_STREAM_MODE":"stateful","ASR_REQUIRE_MODEL_MANIFEST":"true","ASR_EAGER_LOAD":"true","ASR_FILE_TRANSCRIBE_ENABLED":"false","ASR_MODEL_ID":"/models/Qwen3-ASR-1.7B","ASR_MODEL_MANIFEST_PATH":"/models/Qwen3-ASR-1.7B.manifest.json"},"command":["python","-m","uvicorn","app.asr_api:app","--workers","1"]}}}
JSON
  exit 0
fi
if [[ "${1:-}" == compose && "$*" == *' stop qwen-asr-api'* ]]; then
  deployed="$(cat "$FAKE_STATE/deployed")"
  printf 'owner-stop:%s\\n' "$deployed" >>"$FAKE_STATE/events"
  echo stopped >"$FAKE_STATE/deployed"
  exit 0
fi
if [[ "${1:-}" == compose && "$*" == *' up '* ]]; then
  cat "$FAKE_STATE/latest" >"$FAKE_STATE/deployed"
  printf 'owner-start:%s\\n' "$(cat "$FAKE_STATE/deployed")" >>"$FAKE_STATE/events"
  printf 'cutover:%s\\n' "$(cat "$FAKE_STATE/deployed")" >>"$FAKE_STATE/events"
  exit 0
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -eu
printf 'ready:%s\\n' "$(cat "$FAKE_STATE/deployed")" >>"$FAKE_STATE/events"
exit "${FAKE_READY_STATUS:-0}"
""",
    )
    _write_executable(fake_bin / "ffprobe", "#!/usr/bin/env bash\necho 1.0\n")
    _write_executable(fake_bin / "ffmpeg", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        scripts / "verify_asr_release.sh",
        """#!/usr/bin/env bash
set -eu
printf 'verify:%s\\n' "$1" >>"$FAKE_STATE/events"
if [[ "$1" == release ]]; then
  if [[ "$(cat "$FAKE_STATE/deployed")" != stopped ]]; then
    echo 'release attempted while another model owner was running' >&2
    exit 88
  fi
  echo sha256:new >"$FAKE_STATE/latest"
  echo 'ASR release verification passed.'
  exit "${FAKE_RELEASE_STATUS:-0}"
fi
if [[ "$1" != deployed-live ]]; then exit 89; fi
if [[ "$(cat "$FAKE_STATE/deployed")" != sha256:new ]]; then exit 90; fi
echo 'ASR deployed-live verification started.'
exit "${FAKE_LIVE_STATUS:-0}"
""",
    )
    _write_executable(
        scripts / "smoke_asr.sh",
        """#!/usr/bin/env bash
set -eu
deployed="$(cat "$FAKE_STATE/deployed")"
printf 'smoke:%s\\n' "$deployed" >>"$FAKE_STATE/events"
if [[ "$deployed" == sha256:new ]]; then exit "${FAKE_NEW_SMOKE_STATUS:-0}"; fi
old_count_file="$FAKE_STATE/old-smoke-count"
old_count=0
if [[ -f "$old_count_file" ]]; then old_count="$(cat "$old_count_file")"; fi
old_count=$((old_count + 1))
echo "$old_count" >"$old_count_file"
if [[ "$old_count" == 1 ]]; then exit 0; fi
exit "${FAKE_OLD_SMOKE_STATUS:-0}"
""",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_STATE": str(state),
        "FAKE_REPO": str(repo),
        "ASR_RELEASE_ENV_FILE": str(repo / ".env"),
        "ASR_RELEASE_MODEL_DIR": str(model),
        "ASR_RELEASE_MANIFEST": str(
            repo / "models" / "Qwen3-ASR-1.7B.manifest.json"
        ),
        "ASR_DEPLOY_EVIDENCE_DIR": str(secure / "evidence"),
        "ASR_DEPLOY_BACKUP_DIR": str(secure / "backup"),
        "ASR_DEPLOY_LOCAL_BASE_URL": "http://127.0.0.1:8002",
        "ASR_LIVE_BASE_URL": "https://asr.example.internal",
        "ASR_LIVE_WS_URL": "wss://asr.example.internal/v1/transcribe/stream",
        "ASR_LIVE_ZH_AUDIO": str(zh_audio),
        "ASR_LIVE_JA_AUDIO": str(ja_audio),
        "ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS": "10",
        "ASR_LIVE_MAX_GPU_MEMORY_MIB": "23000",
        "ASR_LIVE_GPU_INDEX": "0",
        "ASR_LIVE_API_KEY": "SECRET_SENTINEL_MUST_NOT_LEAK",
        "ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS": "2",
        "ASR_DEPLOY_HEALTH_INTERVAL_SECONDS": "0.01",
    }
    return scripts / RUNNER.name, state, env


def _run_harness(runner: Path, env: dict[str, str]):
    return subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dirty_checkout_fails_closed_without_destructive_git_or_deploy(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_DIRTY"] = "1"
    result = _run_harness(runner, env)
    events = (state / "events").read_text()

    assert result.returncode != 0
    assert "clean Git checkout" in result.stderr
    assert "verify:release" not in events
    assert "cutover:" not in events
    for forbidden in ("git:clean", "git:reset", "git:stash", "git:checkout"):
        assert forbidden not in events


def test_full_workflow_bootstraps_then_releases_cuts_over_and_runs_live(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    result = _run_harness(runner, env)
    events = (state / "events").read_text()

    assert result.returncode == 0, result.stderr
    assert "SECRET_SENTINEL_MUST_NOT_LEAK" not in result.stdout
    assert "SECRET_SENTINEL_MUST_NOT_LEAK" not in result.stderr
    assert "SECRET_SENTINEL_MUST_NOT_LEAK" not in events
    for evidence in (Path(env["ASR_DEPLOY_EVIDENCE_DIR"])).glob("*.log"):
        assert "SECRET_SENTINEL_MUST_NOT_LEAK" not in evidence.read_text()
    assert events.index("python:-m venv") < events.index("verify:release")
    assert events.index("owner-stop:sha256:old") < events.index("verify:release")
    assert events.index("verify:release") < events.index("cutover:sha256:new")
    assert events.index("cutover:sha256:new") < events.index("smoke:sha256:new")
    assert events.index("smoke:sha256:new") < events.index("verify:deployed-live")
    assert "docker:compose build" not in events
    assert "--no-build" in events
    assert "docker:tag sha256:old qwen-asr-api:rollback-" in events
    assert "docker:tag sha256:new qwen-asr-api:release-" in events
    assert (state / "deployed").read_text().strip() == "sha256:new"


def test_release_failure_restores_latest_tag_without_cutover(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_RELEASE_STATUS"] = "19"
    result = _run_harness(runner, env)
    events = (state / "events").read_text()

    assert result.returncode == 19
    assert "cutover:sha256:new" not in events
    assert (state / "latest").read_text().strip() == "sha256:old"
    assert (state / "deployed").read_text().strip() == "sha256:old"


def test_post_cutover_live_failure_rolls_back_and_preserves_failure_status(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_LIVE_STATUS"] = "23"
    result = _run_harness(runner, env)
    events = (state / "events").read_text()

    assert result.returncode == 23
    assert events.index("verify:deployed-live") < events.rindex("cutover:sha256:old")
    assert events.index("verify:deployed-live") < events.rindex("smoke:sha256:old")
    assert (state / "deployed").read_text().strip() == "sha256:old"
    assert "Rollback completed" in result.stderr


def test_rollback_failure_is_reported_without_masking_original_status(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_LIVE_STATUS"] = "23"
    env["FAKE_OLD_SMOKE_STATUS"] = "41"
    result = _run_harness(runner, env)

    assert result.returncode == 23
    assert "ROLLBACK FAILED" in result.stderr
    assert "Original deployment failure status 23 is preserved" in result.stderr
    assert (state / "deployed").read_text().strip() == "sha256:old"


def test_deployed_live_receipt_failure_rolls_back_exact_baseline(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_LIVE_STATUS"] = "31"
    result = _run_harness(runner, env)

    assert result.returncode == 31
    assert "receipt:validate" in (state / "events").read_text()
    assert "Rollback completed" in result.stderr
    assert (state / "deployed").read_text().strip() == "sha256:old"


def test_rollback_never_claims_restoration_with_an_unmatched_baseline_receipt(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_LIVE_STATUS"] = "23"
    env["FAKE_RECEIPT_STATUS"] = "1"
    result = _run_harness(runner, env)

    assert result.returncode == 23
    assert "ROLLBACK FAILED" in result.stderr
    assert "Rollback completed" not in result.stderr
    assert (state / "deployed").read_text().strip() == "stopped"


def test_single_gpu_stops_old_owner_before_release_and_uses_deployed_live(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    result = _run_harness(runner, env)
    events = (state / "events").read_text()

    assert result.returncode == 0, result.stderr
    assert events.index("owner-stop:sha256:old") < events.index("verify:release")
    assert events.index("verify:release") < events.index("cutover:sha256:new")
    assert events.index("cutover:sha256:new") < events.index("verify:deployed-live")
    assert "verify:live" not in events


def test_prechanged_assets_reject_unverified_rollback_baseline(tmp_path):
    runner, state, env = _make_harness(tmp_path)
    env["FAKE_BASELINE_MISMATCH"] = "1"
    result = _run_harness(runner, env)
    events = (state / "events").read_text()

    assert result.returncode != 0
    assert "rollback baseline" in result.stderr.lower()
    assert "verify:release" not in events
    assert "owner-stop:" not in events
