#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime
from pathlib import Path


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_MARKERS = {
    "release": "ASR release verification passed.",
    "rollback-baseline": "ASR rollback baseline verification passed.",
}
PRODUCTION_API_KEY_MIN_LENGTH = 32
PRODUCTION_API_KEY_PLACEHOLDERS = {
    "change-me",
    "replace-with-a-long-random-secret",
    "test-key",
    "your-api-key",
    "your-production-api-key",
    "<your-api-key>",
}


class ReceiptError(RuntimeError):
    pass


def _regular_file(path: str | Path, label: str) -> Path:
    try:
        candidate = Path(path)
        metadata = candidate.lstat()
    except (OSError, TypeError) as exc:
        raise ReceiptError(f"{label} is not an accessible regular file") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ReceiptError(f"{label} must be a regular non-symlink file")
    return candidate.resolve(strict=True)


def _directory(path: str | Path, label: str) -> Path:
    try:
        candidate = Path(path)
        metadata = candidate.lstat()
    except (OSError, TypeError) as exc:
        raise ReceiptError(f"{label} is not an accessible directory") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReceiptError(f"{label} must be a non-symlink directory")
    return candidate.resolve(strict=True)


def _outside_repository(path: Path, repository: Path, label: str) -> None:
    if path == repository or repository in path.parents:
        raise ReceiptError(f"{label} must be outside the repository")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(raw: object, label: str) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if not isinstance(item, str) or "=" not in item:
                raise ReceiptError(f"{label} environment is malformed")
            key, value = item.split("=", 1)
            result[key] = value
        return result
    raise ReceiptError(f"{label} environment is missing")


def _parse_started_at(value: object) -> float:
    if not isinstance(value, str) or not value:
        raise ReceiptError("running rollback container has no start timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ReceiptError("running rollback container start timestamp is invalid") from exc


def validate_baseline(args: argparse.Namespace) -> None:
    compose_path = _regular_file(args.compose_config, "rendered Compose config")
    inspect_path = _regular_file(args.container_inspect, "container inspection")
    try:
        compose = json.loads(compose_path.read_text())
        inspected = json.loads(inspect_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError("rollback baseline Docker metadata is malformed") from exc
    _validate_baseline_payload(args, compose, inspected)


def validate_running_baseline(args: argparse.Namespace) -> None:
    env_file = _regular_file(args.env_file, "rollback environment file")
    compose_result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "config",
            "--format",
            "json",
        ],
        cwd=args.repository,
        text=True,
        capture_output=True,
        check=False,
    )
    inspect_result = subprocess.run(
        ["docker", "inspect", args.container_id],
        text=True,
        capture_output=True,
        check=False,
    )
    hash_result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "config",
            "--hash",
            "qwen-asr-api",
        ],
        cwd=args.repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        compose_result.returncode != 0
        or inspect_result.returncode != 0
        or hash_result.returncode != 0
    ):
        raise ReceiptError("unable to inspect the running rollback baseline")
    try:
        compose = json.loads(compose_result.stdout)
        inspected = json.loads(inspect_result.stdout)
    except json.JSONDecodeError as exc:
        raise ReceiptError("rollback baseline Docker metadata is malformed") from exc
    hash_parts = hash_result.stdout.split()
    if len(hash_parts) != 2 or hash_parts[0] != "qwen-asr-api":
        raise ReceiptError("current Compose service hash is malformed")
    args.compose_hash = hash_parts[1]
    _validate_baseline_payload(args, compose, inspected)


def _validate_baseline_payload(
    args: argparse.Namespace,
    compose: object,
    inspected: object,
) -> None:
    repository = _directory(args.repository, "repository")
    model_dir = _directory(args.model_dir, "rollback model directory")
    manifest = _regular_file(args.manifest, "rollback manifest")
    env_file = _regular_file(args.env_file, "rollback environment file")

    try:
        service = compose["services"]["qwen-asr-api"]
    except (KeyError, TypeError) as exc:
        raise ReceiptError("rollback baseline Docker metadata is malformed") from exc
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise ReceiptError("rollback baseline must inspect exactly one container")
    container = inspected[0]
    if not isinstance(service, dict) or not isinstance(container, dict):
        raise ReceiptError("rollback baseline Docker metadata is malformed")
    try:
        container_config = container["Config"]
        container_state = container["State"]
        image_id = container["Image"]
        mounts = container["Mounts"]
    except (KeyError, TypeError) as exc:
        raise ReceiptError("rollback container inspection is incomplete") from exc
    if (
        not isinstance(container_config, dict)
        or not isinstance(container_state, dict)
        or not isinstance(mounts, list)
    ):
        raise ReceiptError("rollback container inspection is incomplete")
    if container_state.get("Running") is not True or not image_id:
        raise ReceiptError("rollback baseline container must be running")
    labels = container_config.get("Labels") or {}
    if not isinstance(labels, dict) or labels.get(
        "com.docker.compose.config-hash"
    ) != args.compose_hash:
        raise ReceiptError(
            "rollback baseline full Compose service hash does not match"
        )

    service_environment = _environment(service.get("environment"), "Compose")
    container_environment = _environment(container_config.get("Env"), "container")
    mismatched_names = sorted(
        name
        for name, value in service_environment.items()
        if container_environment.get(name) != value
    )
    if mismatched_names:
        raise ReceiptError(
            "rollback baseline environment differs for variable names: "
            + ", ".join(mismatched_names)
        )

    required = {
        "ASR_BACKEND": "qwen_vllm",
        "ASR_STREAM_MODE": "stateful",
        "ASR_REQUIRE_MODEL_MANIFEST": "true",
        "ASR_EAGER_LOAD": "true",
        "ASR_FILE_TRANSCRIBE_ENABLED": "false",
    }
    for name, expected in required.items():
        if container_environment.get(name, "").lower() != expected:
            raise ReceiptError(f"rollback baseline {name} must be {expected}")
    api_key = container_environment.get("API_KEY", "").strip()
    if (
        len(api_key) < PRODUCTION_API_KEY_MIN_LENGTH
        or api_key.lower() in PRODUCTION_API_KEY_PLACEHOLDERS
    ):
        raise ReceiptError("rollback baseline API_KEY is not production strength")

    expected_model = "/models/" + model_dir.relative_to(repository / "models").as_posix()
    expected_manifest = "/models/" + manifest.relative_to(
        repository / "models"
    ).as_posix()
    if container_environment.get("ASR_MODEL_ID") != expected_model:
        raise ReceiptError("rollback baseline ASR_MODEL_ID does not match the model directory")
    if container_environment.get("ASR_MODEL_MANIFEST_PATH") != expected_manifest:
        raise ReceiptError(
            "rollback baseline ASR_MODEL_MANIFEST_PATH does not match the manifest"
        )

    compose_command = [str(item) for item in service.get("command") or []]
    container_command = [str(item) for item in container_config.get("Cmd") or []]
    if compose_command != container_command:
        raise ReceiptError("rollback baseline command differs from current Compose")
    if "--workers" not in container_command:
        raise ReceiptError("rollback baseline command must set one Uvicorn worker")
    worker_index = container_command.index("--workers")
    if (
        worker_index + 1 >= len(container_command)
        or container_command[worker_index + 1] != "1"
    ):
        raise ReceiptError("rollback baseline command must set one Uvicorn worker")

    expected_mount_source = (repository / "models").resolve(strict=True)
    matching_mounts = [
        item
        for item in mounts
        if isinstance(item, dict) and item.get("Destination") == "/models"
    ]
    if len(matching_mounts) != 1:
        raise ReceiptError("rollback baseline must have exactly one /models mount")
    model_mount = matching_mounts[0]
    try:
        actual_mount_source = Path(model_mount["Source"]).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise ReceiptError("rollback baseline /models mount source is invalid") from exc
    if actual_mount_source != expected_mount_source or model_mount.get("RW") is not False:
        raise ReceiptError("rollback baseline /models mount must match and be read-only")

    started_at = _parse_started_at(container_state.get("StartedAt"))
    protected_paths = [env_file, manifest]
    for artifact in model_dir.rglob("*"):
        metadata = artifact.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ReceiptError("rollback model directory contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            protected_paths.append(artifact)
    newer = [path for path in protected_paths if path.stat().st_mtime > started_at]
    if newer:
        raise ReceiptError(
            "rollback config/model/manifest changed after the container started"
        )


def create_receipt(args: argparse.Namespace) -> None:
    if args.kind not in RECEIPT_MARKERS:
        raise ReceiptError("unsupported receipt kind")
    repository = _directory(args.repository, "repository")
    env_file = _regular_file(args.env_file, "release environment file")
    manifest = _regular_file(args.manifest, "release manifest")
    model_dir = _directory(args.model_dir, "release model directory")
    evidence = _regular_file(args.evidence, "verification evidence")
    output = Path(args.output)
    output_parent = _directory(output.parent, "receipt parent")
    _outside_repository(output_parent, repository, "receipt")
    _outside_repository(evidence, repository, "verification evidence")
    if output.exists() or output.is_symlink():
        raise ReceiptError("receipt output already exists")
    if re.fullmatch(r"[0-9a-f]{40}", args.candidate_sha) is None:
        raise ReceiptError("candidate SHA must be a full commit identity")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id) is None:
        raise ReceiptError("image ID must be a Docker sha256 identity")
    marker = RECEIPT_MARKERS[args.kind]
    if marker not in evidence.read_text(errors="replace").splitlines():
        raise ReceiptError("verification evidence does not contain its success marker")

    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": args.kind,
        "candidate_sha": args.candidate_sha,
        "image_id": args.image_id,
        "env_path": str(env_file),
        "env_sha256": _sha256(env_file),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "model_dir": str(model_dir),
        "evidence_path": str(evidence),
        "evidence_sha256": _sha256(evidence),
    }
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")


def validate_receipt(args: argparse.Namespace) -> None:
    receipt = _regular_file(args.receipt, "deployment receipt")
    repository = _directory(args.repository, "repository")
    _outside_repository(receipt, repository, "deployment receipt")
    env_file = _regular_file(args.env_file, "release environment file")
    manifest = _regular_file(args.manifest, "release manifest")
    model_dir = _directory(args.model_dir, "release model directory")
    try:
        payload = json.loads(receipt.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError("deployment receipt is not valid JSON") from exc
    expected_keys = {
        "schema_version",
        "kind",
        "candidate_sha",
        "image_id",
        "env_path",
        "env_sha256",
        "manifest_path",
        "manifest_sha256",
        "model_dir",
        "evidence_path",
        "evidence_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ReceiptError("deployment receipt schema is invalid")
    if not all(
        isinstance(payload[name], str)
        for name in expected_keys - {"schema_version"}
    ) or type(payload["schema_version"]) is not int:
        raise ReceiptError("deployment receipt schema is invalid")
    scalar_checks = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": args.kind,
        "candidate_sha": args.candidate_sha,
        "image_id": args.image_id,
        "env_path": str(env_file),
        "env_sha256": _sha256(env_file),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "model_dir": str(model_dir),
    }
    mismatches = [name for name, expected in scalar_checks.items() if payload[name] != expected]
    if mismatches:
        raise ReceiptError(
            "deployment receipt identity mismatch for fields: " + ", ".join(mismatches)
        )
    evidence = _regular_file(payload["evidence_path"], "receipt evidence")
    _outside_repository(evidence, repository, "receipt evidence")
    if _sha256(evidence) != payload["evidence_sha256"]:
        raise ReceiptError("receipt evidence checksum mismatch")
    marker = RECEIPT_MARKERS.get(args.kind)
    if marker is None or marker not in evidence.read_text(errors="replace").splitlines():
        raise ReceiptError("receipt evidence success marker is missing")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Validate ASR deployment baselines and receipts")
    commands = root.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("validate-baseline")
    baseline.add_argument("--compose-config", required=True)
    baseline.add_argument("--container-inspect", required=True)
    baseline.add_argument("--repository", required=True)
    baseline.add_argument("--compose-hash", required=True)
    baseline.add_argument("--env-file", required=True)
    baseline.add_argument("--model-dir", required=True)
    baseline.add_argument("--manifest", required=True)
    baseline.set_defaults(run=validate_baseline)

    running_baseline = commands.add_parser("validate-running-baseline")
    running_baseline.add_argument("--container-id", required=True)
    running_baseline.add_argument("--repository", required=True)
    running_baseline.add_argument("--env-file", required=True)
    running_baseline.add_argument("--model-dir", required=True)
    running_baseline.add_argument("--manifest", required=True)
    running_baseline.set_defaults(run=validate_running_baseline)

    create = commands.add_parser("create-receipt")
    create.add_argument("--kind", choices=sorted(RECEIPT_MARKERS), required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--repository", required=True)
    create.add_argument("--candidate-sha", required=True)
    create.add_argument("--image-id", required=True)
    create.add_argument("--env-file", required=True)
    create.add_argument("--model-dir", required=True)
    create.add_argument("--manifest", required=True)
    create.add_argument("--evidence", required=True)
    create.set_defaults(run=create_receipt)

    validate = commands.add_parser("validate-receipt")
    validate.add_argument("--kind", choices=sorted(RECEIPT_MARKERS), required=True)
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--repository", required=True)
    validate.add_argument("--candidate-sha", required=True)
    validate.add_argument("--image-id", required=True)
    validate.add_argument("--env-file", required=True)
    validate.add_argument("--model-dir", required=True)
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(run=validate_receipt)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.run(args)
    except (ReceiptError, OSError, ValueError) as exc:
        print(f"ASR deployment receipt validation failed: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
