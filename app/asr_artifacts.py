from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ModelArtifactVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelManifestMetadata:
    source: str
    revision: str
    file_count: int


@dataclass(frozen=True)
class _ManifestEntry:
    path: str
    size: int
    sha256: str


def verify_model_manifest(
    model_dir: str | Path,
    manifest_path: str | Path,
) -> ModelManifestMetadata:
    root = _require_model_directory(model_dir)
    manifest = Path(manifest_path)
    source, revision, entries = _read_manifest(manifest)
    actual_files = _collect_model_files(root, excluded_file=manifest)
    expected_files = {entry.path for entry in entries}
    actual_paths = set(actual_files)
    if actual_paths != expected_files:
        missing = sorted(expected_files - actual_paths)
        unexpected = sorted(actual_paths - expected_files)
        detail = []
        if missing:
            detail.append(f"missing={missing[0]}")
        if unexpected:
            detail.append(f"unexpected={unexpected[0]}")
        raise ModelArtifactVerificationError(
            "Model artifact set does not match manifest: " + ", ".join(detail)
        )

    for entry in entries:
        artifact = actual_files[entry.path]
        size, digest = _measure_file(artifact)
        if size != entry.size:
            raise ModelArtifactVerificationError(
                f"Model artifact size mismatch: {entry.path}"
            )
        if digest != entry.sha256:
            raise ModelArtifactVerificationError(
                f"Model artifact checksum mismatch: {entry.path}"
            )

    return ModelManifestMetadata(source, revision, len(entries))


def create_model_manifest(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    source: str,
    revision: str,
) -> ModelManifestMetadata:
    root = _require_model_directory(model_dir)
    output = Path(output_path)
    output_parent = output.parent.resolve(strict=True)
    resolved_output = output_parent / output.name
    if resolved_output == root or root in resolved_output.parents:
        raise ModelArtifactVerificationError(
            "Create the approval manifest outside the model directory"
        )
    if output.is_symlink():
        raise ModelArtifactVerificationError("Manifest output must not be a symlink")
    source = source.strip()
    revision = revision.strip()
    if not source or not revision:
        raise ModelArtifactVerificationError(
            "Manifest source and immutable revision are required"
        )

    files = _collect_model_files(root)
    entries = []
    for relative_path, artifact in sorted(files.items()):
        size, digest = _measure_file(artifact)
        entries.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": digest,
            }
        )
    if not entries:
        raise ModelArtifactVerificationError("Model directory contains no artifacts")

    payload = {
        "schema_version": 1,
        "source": source,
        "revision": revision,
        "files": entries,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output_parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_name, resolved_output)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return ModelManifestMetadata(source, revision, len(entries))


def _read_manifest(path: Path) -> tuple[str, str, list[_ManifestEntry]]:
    try:
        manifest_bytes = _read_manifest_bytes(path)
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactVerificationError("Model manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "source",
        "revision",
        "files",
    }:
        raise ModelArtifactVerificationError("Model manifest schema is invalid")
    schema_version = payload["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ModelArtifactVerificationError("Model manifest schema is invalid")
    if schema_version != 1:
        raise ModelArtifactVerificationError("Unsupported model manifest schema")
    source = payload["source"]
    revision = payload["revision"]
    files = payload["files"]
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(revision, str)
        or not revision.strip()
        or not isinstance(files, list)
        or not files
    ):
        raise ModelArtifactVerificationError("Model manifest metadata is invalid")

    entries = []
    seen_paths = set()
    for raw_entry in files:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "size",
            "sha256",
        }:
            raise ModelArtifactVerificationError("Model manifest file entry is invalid")
        relative_path = _validate_relative_path(raw_entry["path"])
        size = raw_entry["size"]
        checksum = raw_entry["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(checksum, str)
            or not SHA256_PATTERN.fullmatch(checksum)
        ):
            raise ModelArtifactVerificationError(
                f"Model manifest metadata is invalid: {relative_path}"
            )
        if relative_path in seen_paths:
            raise ModelArtifactVerificationError(
                f"Model manifest contains duplicate path: {relative_path}"
            )
        seen_paths.add(relative_path)
        entries.append(_ManifestEntry(relative_path, size, checksum))
    return source.strip(), revision.strip(), entries


def _read_manifest_bytes(path: Path) -> bytes:
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelArtifactVerificationError("Model manifest does not exist") from exc
    resolved = resolved_parent / path.name
    if not path.name:
        raise ModelArtifactVerificationError("Model manifest must be a regular file")

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None or os.stat not in os.supports_follow_symlinks:
        raise ModelArtifactVerificationError(
            "Safe model manifest verification is unsupported on this platform"
        )
    flags = (
        os.O_RDONLY
        | no_follow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ModelArtifactVerificationError(
            "Unable to open model manifest safely"
        ) from exc

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ModelArtifactVerificationError(
                "Model manifest must be a regular file"
            )
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ModelArtifactVerificationError(
            "Unable to read model manifest safely"
        ) from exc
    finally:
        os.close(descriptor)

    manifest_bytes = b"".join(chunks)
    try:
        current = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ModelArtifactVerificationError(
            "Model manifest changed during verification"
        ) from exc
    if (
        _file_state(before) != _file_state(after)
        or _file_state(after) != _file_state(current)
        or len(manifest_bytes) != after.st_size
    ):
        raise ModelArtifactVerificationError(
            "Model manifest changed during verification"
        )
    return manifest_bytes


def _file_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_model_directory(model_dir: str | Path) -> Path:
    requested_path = Path(model_dir)
    if requested_path.is_symlink():
        raise ModelArtifactVerificationError("ASR model path must not be a symlink")
    try:
        root = requested_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelArtifactVerificationError("ASR model directory does not exist") from exc
    if not root.is_dir() or root.is_symlink():
        raise ModelArtifactVerificationError("ASR model path must be a regular directory")
    return root


def _collect_model_files(
    root: Path,
    *,
    excluded_file: Path | None = None,
) -> dict[str, Path]:
    excluded = excluded_file.resolve() if excluded_file is not None else None
    files = {}
    for artifact in root.rglob("*"):
        if artifact.is_symlink():
            relative_path = artifact.relative_to(root).as_posix()
            raise ModelArtifactVerificationError(
                f"Model artifact must not be a symlink: {relative_path}"
            )
        if artifact.is_dir():
            continue
        if not artifact.is_file():
            relative_path = artifact.relative_to(root).as_posix()
            raise ModelArtifactVerificationError(
                f"Model artifact must be a regular file: {relative_path}"
            )
        resolved = artifact.resolve(strict=True)
        if excluded is not None and resolved == excluded:
            continue
        relative_path = artifact.relative_to(root).as_posix()
        files[relative_path] = artifact
    return files


def _validate_relative_path(value) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelArtifactVerificationError("Model manifest path is invalid")
    relative_path = PurePosixPath(value)
    if (
        relative_path.is_absolute()
        or value != relative_path.as_posix()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ModelArtifactVerificationError("Model manifest path is invalid")
    return value


def _measure_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ModelArtifactVerificationError(
            f"Unable to open model artifact safely: {path.name}"
        ) from exc
    with os.fdopen(descriptor, "rb") as artifact:
        before = os.fstat(artifact.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ModelArtifactVerificationError(
                f"Model artifact must be a regular file: {path.name}"
            )
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(artifact.fileno())
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ModelArtifactVerificationError(
            f"Model artifact changed during verification: {path.name}"
        )
    return after.st_size, digest.hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify an ASR model manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--model-dir", required=True)
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--source", required=True)
    create_parser.add_argument("--revision", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--model-dir", required=True)
    verify_parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "create":
        metadata = create_model_manifest(
            args.model_dir,
            args.output,
            source=args.source,
            revision=args.revision,
        )
        action = "Created manifest for"
    else:
        metadata = verify_model_manifest(args.model_dir, args.manifest)
        action = "Verified"
    print(
        f"{action} {metadata.file_count} model artifacts "
        f"for {metadata.source} at revision {metadata.revision}"
    )


if __name__ == "__main__":
    _main()
