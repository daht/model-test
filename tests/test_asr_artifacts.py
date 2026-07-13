import hashlib
import json

import pytest
from pydantic import ValidationError

import app.asr_artifacts as asr_artifacts
from app.asr import QwenASRTranscriber, QwenVLLMASRTranscriber
from app.asr_artifacts import (
    ModelArtifactVerificationError,
    create_model_manifest,
    verify_model_manifest,
)
from app.config import Settings


def _entry(path, content):
    return {
        "path": path,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_manifest(path, entries, *, schema_version=1):
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "source": "unit-test-model-source",
                "revision": "unit-test-immutable-revision",
                "files": entries,
            }
        )
    )


def test_model_manifest_verifies_exact_local_artifact_set(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = b'{"model_type":"qwen3_asr"}'
    weights = b"test-only-weights"
    (model_dir / "config.json").write_bytes(config)
    (model_dir / "model.safetensors").write_bytes(weights)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            _entry("config.json", config),
            _entry("model.safetensors", weights),
        ],
    )

    metadata = verify_model_manifest(model_dir, manifest)

    assert metadata.source == "unit-test-model-source"
    assert metadata.revision == "unit-test-immutable-revision"
    assert metadata.file_count == 2


def test_trusted_staging_can_create_manifest_then_verify_target_copy(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manifest = tmp_path / "approved-manifest.json"

    created = create_model_manifest(
        model_dir,
        manifest,
        source="unit-test-model-source",
        revision="unit-test-immutable-revision",
    )
    verified = verify_model_manifest(model_dir, manifest)

    assert created == verified
    assert verified.file_count == 1


def test_model_manifest_rejects_symlinked_artifacts(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    approved = tmp_path / "approved.bin"
    approved.write_bytes(b"approved-test-content")
    (model_dir / "model.safetensors").symlink_to(approved)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [_entry("model.safetensors", b"approved-test-content")],
    )

    with pytest.raises(ModelArtifactVerificationError, match="symlink"):
        verify_model_manifest(model_dir, manifest)


def test_model_manifest_rejects_symlink_swap_before_manifest_read(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    content = b"approved-test-content"
    (model_dir / "model.safetensors").write_bytes(content)
    manifest = tmp_path / "manifest.json"
    attacker_manifest = tmp_path / "attacker.json"
    entries = [_entry("model.safetensors", content)]
    _write_manifest(manifest, entries)
    _write_manifest(attacker_manifest, entries)
    original_read_manifest = asr_artifacts._read_manifest

    def swap_then_read(path):
        manifest.unlink()
        manifest.symlink_to(attacker_manifest)
        return original_read_manifest(path)

    monkeypatch.setattr(asr_artifacts, "_read_manifest", swap_then_read)

    with pytest.raises(ModelArtifactVerificationError, match="manifest"):
        verify_model_manifest(model_dir, manifest)


def test_model_manifest_rejects_content_mutation_during_descriptor_read(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    content = b"approved-test-content"
    (model_dir / "model.safetensors").write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [_entry("model.safetensors", content)])
    original_read = asr_artifacts.os.read
    mutated = False

    def read_then_mutate(descriptor, size):
        nonlocal mutated
        data = original_read(descriptor, size)
        if data and not mutated:
            mutated = True
            with manifest.open("ab") as manifest_file:
                manifest_file.write(b" ")
        return data

    monkeypatch.setattr(asr_artifacts.os, "read", read_then_mutate)

    with pytest.raises(ModelArtifactVerificationError, match="changed"):
        verify_model_manifest(model_dir, manifest)


def test_model_manifest_rejects_path_replacement_during_descriptor_read(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    content = b"approved-test-content"
    (model_dir / "model.safetensors").write_bytes(content)
    manifest = tmp_path / "manifest.json"
    replacement = tmp_path / "replacement.json"
    entries = [_entry("model.safetensors", content)]
    _write_manifest(manifest, entries)
    _write_manifest(replacement, entries)
    original_read = asr_artifacts.os.read
    replaced = False

    def read_then_replace(descriptor, size):
        nonlocal replaced
        data = original_read(descriptor, size)
        if data and not replaced:
            replaced = True
            replacement.replace(manifest)
        return data

    monkeypatch.setattr(asr_artifacts.os, "read", read_then_replace)

    with pytest.raises(ModelArtifactVerificationError, match="changed"):
        verify_model_manifest(model_dir, manifest)


@pytest.mark.parametrize("schema_version", [True, False, 1.0, "1", 2])
def test_model_manifest_requires_exact_integer_schema_version(
    tmp_path,
    schema_version,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    content = b"approved-test-content"
    (model_dir / "model.safetensors").write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [_entry("model.safetensors", content)],
        schema_version=schema_version,
    )

    with pytest.raises(ModelArtifactVerificationError, match="schema"):
        verify_model_manifest(model_dir, manifest)


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_model_manifest_fails_closed_on_artifact_set_or_digest_changes(
    tmp_path,
    mutation,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    original = b"approved-test-content"
    artifact = model_dir / "model.safetensors"
    artifact.write_bytes(original)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [_entry("model.safetensors", original)])

    if mutation == "changed":
        artifact.write_bytes(b"changed-test-content")
    elif mutation == "extra":
        (model_dir / "unapproved.json").write_text("{}")
    else:
        artifact.unlink()

    with pytest.raises(ModelArtifactVerificationError):
        verify_model_manifest(model_dir, manifest)


@pytest.mark.parametrize("manifest_path", [None, "", "   "])
def test_required_model_manifest_setting_needs_operator_path(manifest_path):
    with pytest.raises(ValidationError, match="manifest path"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            api_key="unit-test-only-not-a-production-secret-000000",
            asr_require_model_manifest=True,
            asr_model_manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("transcriber_type", "backend", "stream_mode"),
    [
        (QwenASRTranscriber, "qwen", "chunked"),
        (QwenVLLMASRTranscriber, "qwen_vllm", "stateful"),
    ],
)
def test_qwen_startup_verifies_operator_manifest_before_loading_runtime(
    tmp_path,
    transcriber_type,
    backend,
    stream_mode,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    artifact = model_dir / "model.safetensors"
    artifact.write_bytes(b"tampered-test-content")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [_entry("model.safetensors", b"approved-test-content")],
    )
    settings = Settings(
        _env_file=None,
        asr_backend=backend,
        asr_stream_mode=stream_mode,
        asr_model_id=str(model_dir),
        api_key="unit-test-only-not-a-production-secret-000000",
        asr_require_model_manifest=True,
        asr_model_manifest_path=str(manifest),
    )

    with pytest.raises(ModelArtifactVerificationError):
        transcriber_type(settings).warmup()
