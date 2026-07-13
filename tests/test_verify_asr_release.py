import os
import subprocess
from pathlib import Path


RUNNER = Path("scripts/verify_asr_release.sh")


def run_runner(*args, env=None):
    runner_env = os.environ.copy()
    if env:
        runner_env.update(env)
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        env=runner_env,
        text=True,
        capture_output=True,
        timeout=10,
    )


def test_release_runner_help_and_gate_inventory_are_actionable():
    help_result = run_runner("--help")
    list_result = run_runner("--list-gates", "live")

    assert help_result.returncode == 0
    assert "Usage:" in help_result.stdout
    assert "commit" in help_result.stdout
    assert "release" in help_result.stdout
    assert "live" in help_result.stdout
    assert list_result.returncode == 0
    assert "C01" in list_result.stdout
    assert "R01" in list_result.stdout
    assert "L01" in list_result.stdout
    assert "full explicit-mock pytest" in list_result.stdout


def test_release_runner_rejects_unknown_mode():
    result = run_runner("unknown")

    assert result.returncode == 2
    assert "Unknown mode" in result.stderr


def test_commit_dry_run_plans_full_suite_once_and_all_local_gates():
    result = run_runner("--dry-run", "commit")

    assert result.returncode == 0
    assert result.stdout.count("pytest tests -q") == 1
    assert "compileall" in result.stdout
    assert "bash -n" in result.stdout
    assert "git diff --cached --check" in result.stdout
    assert "high-confidence secret scan" in result.stdout
    assert "forbidden tracked paths" in result.stdout


def test_live_dry_run_includes_release_and_strict_language_chunk_matrix():
    result = run_runner("live", "--dry-run")

    assert result.returncode == 0
    assert "docker compose" in result.stdout
    assert "real Qwen warmup" in result.stdout
    assert "scripts/smoke_asr.sh" in result.stdout
    assert "--language zh --chunk-ms 200" in result.stdout
    assert "--language zh --chunk-ms 500" in result.stdout
    assert "--language ja --chunk-ms 200" in result.stdout
    assert "--language ja --chunk-ms 500" in result.stdout
    assert "--verify-protocol" in result.stdout
    assert "concurrent zh+ja" in result.stdout
    assert "GPU memory" in result.stdout


def test_release_mode_missing_prerequisites_fails_before_commit_suite(tmp_path):
    result = run_runner(
        "release",
        env={
            "ASR_RELEASE_ENV_FILE": str(tmp_path / "missing.env"),
            "ASR_RELEASE_MODEL_DIR": str(tmp_path / "missing-model"),
            "ASR_RELEASE_MANIFEST": str(tmp_path / "missing-manifest.json"),
        },
    )

    assert result.returncode == 1
    assert "Missing release prerequisites" in result.stderr
    assert "full explicit-mock pytest" not in result.stdout


def test_live_mode_never_echoes_runtime_secret_on_preflight_failure(tmp_path):
    dummy_secret = "test-only-runtime-secret-never-print-12345"
    result = run_runner(
        "live",
        env={
            "ASR_LIVE_API_KEY": dummy_secret,
            "ASR_RELEASE_ENV_FILE": str(tmp_path / "missing.env"),
        },
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "Missing live prerequisites" in output
    assert dummy_secret not in output


def test_help_and_dry_run_leave_repo_and_temp_directory_unchanged(tmp_path):
    before_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    before_temp = sorted(tmp_path.iterdir())

    help_result = run_runner("--help", env={"TMPDIR": str(tmp_path)})
    dry_result = run_runner("commit", "--dry-run", env={"TMPDIR": str(tmp_path)})

    after_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert help_result.returncode == 0
    assert dry_result.returncode == 0
    assert after_status == before_status
    assert sorted(tmp_path.iterdir()) == before_temp


def test_operator_guide_is_linked_and_documents_fail_closed_layers():
    guide = Path("docs/asr-release-verification.md").read_text()
    readme = Path("README.md").read_text()
    cloud_runbook = Path("cloud/README-A10.md").read_text()

    for subject in (
        "commit mode",
        "release mode",
        "live mode",
        "fresh",
        "rollback",
        "evidence",
        "runtime secret",
    ):
        assert subject in guide.lower()
    assert "scripts/verify_asr_release.sh" in guide
    assert "ASR_LIVE_API_KEY" in guide
    assert "docs/asr-release-verification.md" in readme
    assert "docs/asr-release-verification.md" in cloud_runbook
