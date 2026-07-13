#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Callable


class MonitorValidationError(RuntimeError):
    pass


def query_memory_mib(gpu_index: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=2,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited with status {completed.returncode}")
    lines = completed.stdout.splitlines()
    if not lines:
        raise RuntimeError("nvidia-smi returned no memory value")
    try:
        value = int(lines[0].strip())
    except ValueError as exc:
        raise RuntimeError("nvidia-smi returned an invalid memory value") from exc
    if value < 0:
        raise RuntimeError("nvidia-smi returned a negative memory value")
    return value


def record_samples(
    state_dir: Path,
    *,
    gpu_index: int,
    interval_seconds: float,
    sample_memory: Callable[[int], int] = query_memory_mib,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, object]:
    state_dir.mkdir(parents=True, exist_ok=True)
    stop_file = state_dir / "stop"
    stop_requested = should_stop or stop_file.exists
    started = monotonic()
    attempts = 0
    values: list[int] = []
    sample_times: list[float] = []
    failure_times: list[float] = []

    while not stop_requested():
        sampled_at = monotonic() - started
        attempts += 1
        try:
            value = sample_memory(gpu_index)
        except Exception:
            failure_times.append(sampled_at)
        else:
            values.append(value)
            sample_times.append(sampled_at)
        sleep(interval_seconds)

    finished = monotonic() - started
    sample_span = sample_times[-1] - sample_times[0] if len(sample_times) > 1 else 0.0
    summary: dict[str, object] = {
        "attempts": attempts,
        "successes": len(values),
        "failures": len(failure_times),
        "failure_times": failure_times,
        "sample_times": sample_times,
        "sample_span_seconds": sample_span,
        "monitor_duration_seconds": finished,
        "maximum_memory_mib": max(values) if values else None,
        "interval_seconds": interval_seconds,
        "gpu_index": gpu_index,
    }
    temporary = state_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, sort_keys=True) + "\n")
    temporary.replace(state_dir / "summary.json")
    return summary


def validate_summary(
    summary: dict[str, object],
    *,
    maximum_memory_mib: int,
    minimum_samples: int,
    minimum_span_seconds: float,
) -> None:
    attempts = int(summary.get("attempts", -1))
    successes = int(summary.get("successes", -1))
    failures = int(summary.get("failures", -1))
    sample_span = float(summary.get("sample_span_seconds", -1.0))
    observed_maximum = summary.get("maximum_memory_mib")
    problems = []
    if attempts < 0 or successes < 0 or failures < 0 or attempts != successes + failures:
        problems.append("inconsistent sampling counts")
    if failures:
        problems.append(f"{failures} sampling failures")
    if successes < minimum_samples:
        problems.append(
            f"only {successes} valid samples; at least {minimum_samples} required"
        )
    if sample_span < minimum_span_seconds:
        problems.append(
            f"sample span {sample_span:.3f}s; at least {minimum_span_seconds:.3f}s required"
        )
    if observed_maximum is None:
        problems.append("no valid GPU memory maximum")
    elif int(observed_maximum) > maximum_memory_mib:
        problems.append(
            f"GPU memory {observed_maximum} MiB exceeds {maximum_memory_mib} MiB"
        )
    if problems:
        raise MonitorValidationError("; ".join(problems))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal ASR live GPU sampler.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--state-dir", type=Path, required=True)
    record.add_argument("--gpu-index", type=int, required=True)
    record.add_argument("--interval-seconds", type=float, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--state-dir", type=Path, required=True)
    validate.add_argument("--maximum-memory-mib", type=int, required=True)
    validate.add_argument("--minimum-samples", type=int, required=True)
    validate.add_argument("--minimum-span-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "record":
        record_samples(
            args.state_dir,
            gpu_index=args.gpu_index,
            interval_seconds=args.interval_seconds,
        )
        return
    summary_path = args.state_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit("GPU monitor summary is missing")
    summary = json.loads(summary_path.read_text())
    try:
        validate_summary(
            summary,
            maximum_memory_mib=args.maximum_memory_mib,
            minimum_samples=args.minimum_samples,
            minimum_span_seconds=args.minimum_span_seconds,
        )
    except MonitorValidationError as exc:
        raise SystemExit(f"GPU monitor validation failed: {exc}") from None
    print(
        "GPU monitor passed: "
        f"attempts={summary['attempts']} successes={summary['successes']} "
        f"failures={summary['failures']} span={summary['sample_span_seconds']:.3f}s "
        f"peak={summary['maximum_memory_mib']}MiB"
    )


if __name__ == "__main__":
    main()
