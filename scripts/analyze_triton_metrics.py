#!/usr/bin/env python3
"""Summarize Triton per-model batching from monitor Prometheus snapshots."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


METRICS = {
    "nv_inference_request_success": "successful_requests",
    "nv_inference_exec_count": "executions",
    "nv_inference_count": "inference_elements",
}
SAMPLE_MARKER = "# sampled_at="
LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class Snapshot:
    sampled_at: str
    values: dict[tuple[str, str], float]


def parse_snapshots(source: Path) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    sampled_at: str | None = None
    values: dict[tuple[str, str], float] = {}

    def commit() -> None:
        nonlocal values
        if sampled_at is not None and values:
            snapshots.append(Snapshot(sampled_at, values))
        values = {}

    for raw in source.read_text(encoding="utf-8").splitlines():
        if raw.startswith(SAMPLE_MARKER):
            commit()
            sampled_at = raw.removeprefix(SAMPLE_MARKER).strip()
            continue
        match = LINE.match(raw)
        if match is None or match["name"] not in METRICS:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = dict(LABEL.findall(match["labels"] or ""))
        model = labels.get("model")
        if model is None:
            continue
        values[(model, METRICS[match["name"]])] = value
    commit()
    return snapshots


def summarize(snapshots: list[Snapshot]) -> dict[str, object]:
    series: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        for key, value in snapshot.values.items():
            series[key].append(value)

    models = sorted({model for model, _ in series})
    result: list[dict[str, object]] = []
    for model in models:
        deltas: dict[str, float | None] = {}
        for metric in METRICS.values():
            values = series.get((model, metric), [])
            delta = values[-1] - values[0] if len(values) >= 2 else None
            deltas[metric] = delta if delta is None or delta >= 0 else None
        executions = deltas["executions"]
        requests = deltas["successful_requests"]
        elements = deltas["inference_elements"]
        result.append(
            {
                "model": model,
                "successful_requests_delta": requests,
                "executions_delta": executions,
                "inference_elements_delta": elements,
                "requests_per_execution": ratio(requests, executions),
                "average_batch_size": ratio(elements, executions),
            }
        )
    return {"snapshot_count": len(snapshots), "models": result}


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def render_markdown(report: dict[str, object]) -> str:
    lines = ["# Triton batching evidence", "", f"- Metrics snapshots: {report['snapshot_count']}", ""]
    models = report["models"]
    if not models:
        lines.append("No Triton inference counters were found. Confirm Triton metrics are enabled.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| model | successful requests | executions | inference elements | requests/execution | avg batch size |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in models:
        lines.append(
            "| {model} | {requests} | {executions} | {elements} | {requests_per_execution} | {average_batch_size} |".format(
                model=model["model"],
                requests=format_number(model["successful_requests_delta"]),
                executions=format_number(model["executions_delta"]),
                elements=format_number(model["inference_elements_delta"]),
                requests_per_execution=format_number(model["requests_per_execution"]),
                average_batch_size=format_number(model["average_batch_size"]),
            )
        )
    lines.extend(
        [
            "",
            "`avg batch size = nv_inference_count delta / nv_inference_exec_count delta`. "
            "A value near 1 means that model executed essentially one item at a time.",
            "",
        ]
    )
    return "\n".join(lines)


def format_number(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="triton-metrics.prom collected by monitor_tts_benchmark.sh")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    report = summarize(parse_snapshots(args.metrics))
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
