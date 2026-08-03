#!/usr/bin/env python3
"""Record comparable strict streaming ASR evaluations for two local samples."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

if __package__:
    from scripts import stream_asr_client
else:
    import stream_asr_client


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampleResult:
    language: str
    file: dict[str, object]
    events: list[dict[str, object]]
    transcript: str
    elapsed_seconds: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and record one strict Chinese/English streaming ASR comparison."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ASR_EVAL_WS_URL", os.environ.get("WS_URL")),
        help="ASR WebSocket endpoint (defaults to ASR_EVAL_WS_URL or WS_URL).",
    )
    parser.add_argument("--zh-audio", type=Path, help="Chinese audio sample path")
    parser.add_argument("--en-audio", type=Path, help="English audio sample path")
    parser.add_argument(
        "--output", type=Path, help="Markdown document to append after both streams pass"
    )
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print authenticated stream-info without streaming audio or writing a report.",
    )
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("--url or ASR_EVAL_WS_URL/WS_URL is required")
    if args.chunk_ms < 100 or args.chunk_ms > 500:
        parser.error("--chunk-ms must be between 100 and 500")
    if not args.inspect_only:
        if args.zh_audio is None or args.en_audio is None or args.output is None:
            parser.error("--zh-audio, --en-audio, and --output are required")
    return args


def sanitized_endpoint(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def redact_sensitive(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if str(key).lower() in {"api_key", "authorization", "token"}
            else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def backend_snapshot_url(ws_url: str) -> str:
    stream_info_url = stream_asr_client.default_stream_info_url(ws_url)
    return stream_info_url.rsplit("/v1/transcribe/stream-info", 1)[0] + "/v1/asr/backends"


def fetch_backend_snapshot(ws_url: str, api_key: str) -> dict[str, object]:
    request = urllib.request.Request(
        backend_snapshot_url(ws_url), headers={"X-API-Key": api_key}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationError("backends response must be a JSON object")
    return payload


def model_identity(backends: dict[str, object]) -> dict[str, str]:
    workers = backends.get("workers")
    if not isinstance(workers, dict):
        raise EvaluationError("backends response does not contain workers")
    accepting = [worker for worker in workers.values() if isinstance(worker, dict) and worker.get("accepting") is True]
    if len(accepting) != 1:
        raise EvaluationError("backends response must contain exactly one accepting worker")
    worker = accepting[0]
    model_id = worker.get("model_id")
    revision = worker.get("model_revision")
    gpu_id = worker.get("gpu_id")
    worker_id = worker.get("worker_id")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (model_id, revision, gpu_id, worker_id)
    ):
        raise EvaluationError(
            "accepting worker does not expose model_id, model_revision, gpu_id, and worker_id"
        )
    return {
        "model_id": model_id,
        "model_revision": revision,
        "gpu_id": gpu_id,
        "worker_id": worker_id,
    }


def file_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise EvaluationError(f"audio sample is not a file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as audio:
        for chunk in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def transcript_from_events(events: list[dict[str, object]]) -> str:
    confirmed = [
        str(event.get("text", ""))
        for event in events
        if event.get("type") == "sentence_final"
    ]
    final = next(
        (str(event.get("text", "")) for event in reversed(events) if event.get("type") == "final"),
        "",
    )
    return "".join(confirmed) + final


async def evaluate_sample(
    *,
    path: Path,
    language: str,
    args: argparse.Namespace,
    api_key: str,
    protocol: str,
) -> SampleResult:
    metadata = file_metadata(path)
    client_args = argparse.Namespace(
        api_key=api_key,
        api_key_source="environment",
        audio_file=str(path),
        chunk_ms=args.chunk_ms,
        language=language,
        print_mode="events",
        protocol=protocol,
        realtime=args.realtime,
        sample_rate=16000,
        show_stream_info=False,
        stream_info_url=None,
        url=args.url,
        verify_protocol=True,
    )
    started = time.monotonic()
    try:
        result = await stream_asr_client.stream_audio(client_args)
    except (stream_asr_client.StreamClientError, OSError) as exc:
        raise EvaluationError(f"{language} stream failed") from exc
    elapsed_seconds = time.monotonic() - started
    return SampleResult(
        language=language,
        file=metadata,
        events=result.events,
        transcript=transcript_from_events(result.events),
        elapsed_seconds=elapsed_seconds,
    )


def markdown_block(
    *,
    timestamp: str,
    endpoint: str,
    stream_info: dict[str, object],
    backends: dict[str, object],
    identity: dict[str, str],
    samples: list[SampleResult],
) -> str:
    lines = [
        f"## ASR evaluation {timestamp}",
        "",
        f"- Endpoint: `{endpoint}`",
        f"- Model identity: `{identity['model_id']}` ({identity['model_revision']})",
        f"- Worker: `{identity['worker_id']}` on GPU `{identity['gpu_id']}`",
        "",
        "### Stream Info",
        "",
        "```json",
        json.dumps(stream_info, ensure_ascii=False, indent=2),
        "```",
        "",
        "### Backends",
        "",
        "```json",
        json.dumps(backends, ensure_ascii=False, indent=2),
        "```",
    ]
    for sample in samples:
        lines.extend(
            [
                "",
                f"### {sample.language}",
                "",
                f"- File: `{sample.file['name']}` ({sample.file['bytes']} bytes)",
                f"- SHA256: `{sample.file['sha256']}`",
                f"- Elapsed: {sample.elapsed_seconds:.3f} seconds",
                f"- Final transcript: {sample.transcript}",
                "",
                "#### Events",
                "",
                "```json",
                json.dumps(sample.events, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def append_report(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as report:
        report.write(content)


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise EvaluationError("API_KEY environment variable is required")
    endpoint = sanitized_endpoint(args.url)
    try:
        stream_info = stream_asr_client.fetch_stream_info(
            stream_asr_client.default_stream_info_url(args.url), api_key
        )
    except OSError as exc:
        raise EvaluationError("could not fetch authenticated stream-info") from exc
    if not isinstance(stream_info, dict):
        raise EvaluationError("stream-info response must be a JSON object")
    try:
        backends = fetch_backend_snapshot(args.url, api_key)
    except OSError as exc:
        raise EvaluationError("could not fetch authenticated backends") from exc
    identity = model_identity(backends)
    report_stream_info = redact_sensitive(stream_info)
    report_backends = redact_sensitive(backends)
    assert isinstance(report_stream_info, dict)
    assert isinstance(report_backends, dict)
    if args.inspect_only:
        print(f"Endpoint: {endpoint}")
        print(
            json.dumps(
                {
                    "model_identity": identity,
                    "stream_info": report_stream_info,
                    "backends": report_backends,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    protocol = stream_asr_client.detect_protocol(stream_info)
    samples = [
        await evaluate_sample(
            path=args.zh_audio,
            language="zh",
            args=args,
            api_key=api_key,
            protocol=protocol,
        ),
        await evaluate_sample(
            path=args.en_audio,
            language="en",
            args=args,
            api_key=api_key,
            protocol=protocol,
        ),
    ]
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    append_report(
        args.output,
        markdown_block(
            timestamp=timestamp,
            endpoint=endpoint,
            stream_info=report_stream_info,
            backends=report_backends,
            identity=identity,
            samples=samples,
        ),
    )
    print(f"Recorded {identity['model_id']} evaluation in {args.output}")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except EvaluationError as exc:
        print(f"ASR evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
