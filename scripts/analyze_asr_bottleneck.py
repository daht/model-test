#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _json_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    records = []
    malformed = 0
    for line in path.read_text(errors="replace").splitlines():
        candidate = line[line.find("{") :] if "{" in line else ""
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            records.append(value)
    return records, malformed


def _event_records(run: Path) -> tuple[list[dict[str, Any]], int]:
    events, malformed = _json_records(run / "events.jsonl")
    if events:
        return events, malformed
    return _json_records(run / "asr-service.log")


def _preceding_engine_call(
    event: dict[str, Any], engine_calls: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    event_time = _timestamp(str(event["timestamp"]))
    candidates = []
    for call in engine_calls:
        try:
            age = (event_time - _timestamp(str(call["timestamp"]))).total_seconds()
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= age <= 30:
            candidates.append((age, call))
    if not candidates:
        return None
    _, selected = min(candidates, key=lambda item: item[0])
    return {
        key: selected[key]
        for key in (
            "timestamp",
            "batch_id",
            "engine_call_id",
            "elapsed_seconds",
            "group_size",
            "beam_size",
            "accumulated_audio_max_seconds",
            "maximum_character_run",
        )
        if key in selected
    }


def _csv_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _nearest_sample(event: dict[str, Any], samples: list[dict[str, str]]) -> dict[str, str] | None:
    try:
        event_time = _timestamp(str(event["timestamp"]))
    except (KeyError, TypeError, ValueError):
        return None
    candidates = []
    for sample in samples:
        try:
            distance = abs((event_time - _timestamp(sample["sampled_at"])).total_seconds())
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((distance, sample))
    if not candidates:
        return None
    distance, selected = min(candidates, key=lambda item: item[0])
    return selected if distance <= 2 else None


def analyze_run(run: Path) -> dict[str, Any]:
    run = run.resolve(strict=True)
    metadata = {}
    if (run / "metadata.json").is_file():
        metadata = json.loads((run / "metadata.json").read_text())
    events, malformed = _event_records(run)
    gpu_samples = _csv_records(run / "gpu.csv")
    gpu_process_samples = _csv_records(run / "gpu-processes.csv")
    docker_samples = _csv_records(run / "docker-stats.csv")
    warnings = []
    if not events:
        warnings.append("events.jsonl is missing or empty")
    if malformed:
        warnings.append(f"malformed structured event lines: {malformed}")

    ordered = sorted(events, key=lambda item: str(item.get("timestamp", "")))
    engine_calls = [
        item for item in ordered if item.get("event") == "asr_engine_group_completed"
    ]
    engine_failures = [
        {
            name: item.get(name)
            for name in (
                "timestamp",
                "batch_id",
                "group_ordinal",
                "group_count",
                "group_size",
                "final_items",
                "accumulated_audio_seconds",
                "min_input_audio_seconds",
                "max_input_audio_seconds",
                "failure_stage",
                "exception_type",
            )
        }
        for item in ordered
        if item.get("event") == "asr_engine_group_failed"
    ]
    slowest = sorted(
        engine_calls,
        key=lambda item: float(item.get("elapsed_seconds", 0)),
        reverse=True,
    )[:20]
    opened = {
        str(item["session_id"])
        for item in ordered
        if item.get("event") == "asr_session_opened" and item.get("session_id")
    }
    terminal = {
        str(item["session_id"])
        for item in ordered
        if item.get("event") == "asr_session_terminal" and item.get("session_id")
    }
    released = {
        str(item["session_id"])
        for item in ordered
        if item.get("event") == "asr_session_released" and item.get("session_id")
    }
    rejections = []
    for item in ordered:
        if item.get("event") != "asr_buffer_rejected":
            continue
        record = dict(item)
        record["preceding_engine_call"] = _preceding_engine_call(item, engine_calls)
        record["nearest_gpu_sample"] = _nearest_sample(item, gpu_samples)
        rejections.append(record)

    batches = [item for item in ordered if item.get("event") == "asr_batch_dispatched"]
    return {
        "metadata": metadata,
        "quality": {
            "warnings": warnings,
            "event_count": len(events),
            "malformed_lines": malformed,
        },
        "lifecycle": {
            "opened_sessions": len(opened),
            "terminal_sessions": len(terminal),
            "released_sessions": len(released),
            "incomplete_sessions": sorted(opened - terminal | terminal - released),
        },
        "scheduler": {
            "dispatched_batches": len(batches),
            "largest_batch": max(
                (int(item.get("selected_jobs", 0)) for item in batches), default=0
            ),
        },
        "engine": {
            "calls": len(engine_calls),
            "slowest_calls": slowest,
            "failures": engine_failures,
        },
        "failures": {
            "buffer_rejections": rejections,
            "cleanup_conflicts": [
                item for item in ordered if item.get("event") == "asr_cleanup_conflict"
            ],
        },
        "resources": {
            "gpu": {
                "samples": len(gpu_samples),
                "max_utilization_percent": max(
                    (float(item.get("gpu_util_percent", 0)) for item in gpu_samples),
                    default=0,
                ),
            },
            "gpu_process_samples": len(gpu_process_samples),
            "docker_samples": len(docker_samples),
            "containers": sorted(
                {item["container"] for item in docker_samples if item.get("container")}
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    quality = report["quality"]
    lifecycle = report["lifecycle"]
    scheduler = report["scheduler"]
    engine = report["engine"]
    failures = report["failures"]
    lines = [
        "# ASR Bottleneck Report",
        "",
        f"- Run: `{report['metadata'].get('run_id', 'unknown')}`",
        f"- Structured events: {quality['event_count']}",
        f"- Sessions opened/terminal/released: {lifecycle['opened_sessions']}/{lifecycle['terminal_sessions']}/{lifecycle['released_sessions']}",
        f"- Scheduler batches: {scheduler['dispatched_batches']} (largest {scheduler['largest_batch']})",
        f"- Engine calls: {engine['calls']}",
        f"- Engine failures: {len(engine['failures'])}",
        f"- Buffer rejections: {len(failures['buffer_rejections'])}",
        f"- Cleanup conflicts: {len(failures['cleanup_conflicts'])}",
        "",
        "## Evidence quality",
        "",
    ]
    lines.extend(
        [f"- WARNING: {warning}" for warning in quality["warnings"]]
        or ["- No structured-event parsing warnings."]
    )
    lines.extend(["", "## Slowest engine calls", ""])
    if engine["slowest_calls"]:
        for call in engine["slowest_calls"]:
            lines.append(
                f"- {call.get('timestamp', 'unknown')}: {float(call.get('elapsed_seconds', 0)):.3f}s, "
                f"group={call.get('group_size', 'unknown')}, batch={call.get('batch_id', 'unknown')}"
            )
    else:
        lines.append("- No engine completion events.")
    lines.extend(["", "## Engine failures", ""])
    if engine["failures"]:
        for failure in engine["failures"]:
            lines.append(
                f"- {failure.get('timestamp', 'unknown')}: "
                f"stage={failure.get('failure_stage', 'unknown')}, "
                f"exception={failure.get('exception_type', 'unknown')}, "
                f"group={failure.get('group_size', 'unknown')}, "
                f"batch={failure.get('batch_id', 'unknown')}, "
                f"audio={failure.get('accumulated_audio_seconds', 'unknown')}s"
            )
    else:
        lines.append("- No engine failure events.")
    lines.extend(["", "## Buffer rejections", ""])
    if failures["buffer_rejections"]:
        for rejection in failures["buffer_rejections"]:
            preceding = rejection.get("preceding_engine_call") or {}
            lines.append(
                f"- {rejection.get('timestamp', 'unknown')}: {rejection.get('reason', 'unknown')}, "
                f"current={rejection.get('current', 'unknown')}, limit={rejection.get('limit', 'unknown')}, "
                f"preceding_engine={preceding.get('elapsed_seconds', 'none')}s"
            )
    else:
        lines.append("- No buffer rejection events.")
    lines.append("")
    return "\n".join(lines)


def write_reports(run: Path, report: dict[str, Any]) -> None:
    (run / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (run / "report.md").write_text(_markdown(report))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one ASR monitor run")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    report = analyze_run(args.run_directory)
    write_reports(args.run_directory, report)
    print(args.run_directory / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
