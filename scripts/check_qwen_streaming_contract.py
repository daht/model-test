#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import fields
from importlib.metadata import version


def require_version(distribution: str, expected: str) -> None:
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(
            f"{distribution} contract mismatch: expected {expected}, got {actual}"
        )


def main() -> None:
    require_version("qwen-asr", "0.0.6")
    require_version("vllm", "0.14.0")

    from qwen_asr.inference.qwen3_asr import ASRStreamingState, Qwen3ASRModel

    required_fields = {
        "chunk_size_samples",
        "chunk_id",
        "buffer",
        "audio_accum",
        "language",
        "text",
    }
    actual_fields = {field.name for field in fields(ASRStreamingState)}
    missing = required_fields - actual_fields
    if missing:
        raise SystemExit(
            "qwen-asr ASRStreamingState is missing fields: "
            + ", ".join(sorted(missing))
        )

    contracts = {
        "init_streaming_state": {"language", "chunk_size_sec"},
        "streaming_transcribe": {"pcm16k", "state"},
        "finish_streaming_transcribe": {"state"},
    }
    for method_name, required_parameters in contracts.items():
        method = getattr(Qwen3ASRModel, method_name)
        parameters = set(inspect.signature(method).parameters)
        missing_parameters = required_parameters - parameters
        if missing_parameters:
            raise SystemExit(
                f"qwen-asr {method_name} is missing parameters: "
                + ", ".join(sorted(missing_parameters))
            )


if __name__ == "__main__":
    main()
