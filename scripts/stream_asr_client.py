#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

import websockets


class DisplayState:
    def __init__(self) -> None:
        self.confirmed: list[str] = []
        self.tail = ""

    def apply(self, payload: dict[str, object]) -> str | None:
        message_type = payload.get("type")
        text = str(payload.get("text", ""))
        if message_type == "sentence_final":
            self.confirmed.append(text)
            self.tail = ""
        elif message_type in {"partial", "final"}:
            self.tail = text
        else:
            return None
        return "".join(self.confirmed) + self.tail


class SequenceTracker:
    def __init__(self) -> None:
        self.last_sequence: int | None = None

    def observe(self, payload: dict[str, object]) -> str | None:
        sequence = payload.get("sequence")
        if not isinstance(sequence, int):
            return None
        previous = self.last_sequence
        self.last_sequence = sequence
        if previous is None:
            return None
        if sequence <= previous:
            return f"server event sequence is not increasing: {sequence} after {previous}"
        if sequence != previous + 1:
            return f"server event sequence gap: expected {previous + 1}, got {sequence}"
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Qwen ASR WebSocket streaming.")
    parser.add_argument("audio_file", help="Input audio file, such as wav/mp3/m4a/flac")
    parser.add_argument(
        "--url",
        default=os.environ.get("WS_URL", "ws://127.0.0.1:8002/v1/transcribe/stream"),
        help="WebSocket endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key, defaults to API_KEY env var",
    )
    parser.add_argument("--language", default=os.environ.get("LANGUAGE", "zh"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--print-mode", choices=["events", "display"], default="events")
    parser.add_argument("--stream-info-url", default=os.environ.get("STREAM_INFO_URL"))
    parser.add_argument("--show-stream-info", action="store_true")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between chunks to simulate microphone streaming.",
    )
    return parser.parse_args()


def default_stream_info_url(ws_url: str) -> str:
    if ws_url.startswith("wss://"):
        base = "https://" + ws_url[len("wss://") :]
    elif ws_url.startswith("ws://"):
        base = "http://" + ws_url[len("ws://") :]
    else:
        return ws_url
    return base.rsplit("/v1/transcribe/stream", 1)[0] + "/v1/transcribe/stream-info"


def fetch_stream_info(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def start_ffmpeg(audio_file: str, sample_rate: int) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_file,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE)


async def receive_messages(
    websocket: websockets.ClientConnection,
    print_mode: str = "events",
    sequence_tracker: SequenceTracker | None = None,
) -> None:
    display_state = DisplayState()
    tracker = sequence_tracker or SequenceTracker()
    async for message in websocket:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print(message)
            continue

        if warning := tracker.observe(payload):
            print(f"warning: {warning}", file=sys.stderr)

        message_type = payload.get("type")
        if print_mode == "display":
            display_text = display_state.apply(payload)
            if display_text is not None:
                print(f"[display] {display_text}")
            else:
                print(json.dumps(payload, ensure_ascii=False))
        elif message_type in {"partial", "sentence_final", "final"}:
            print(f"[{message_type}] {payload.get('text', '')}")
        else:
            print(json.dumps(payload, ensure_ascii=False))


async def stream_audio(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set API_KEY.")

    if args.show_stream_info:
        stream_info_url = args.stream_info_url or default_stream_info_url(args.url)
        print("ASR stream info:")
        print(json.dumps(fetch_stream_info(stream_info_url), ensure_ascii=False, indent=2))

    bytes_per_second = args.sample_rate * 2
    chunk_size = max(1, bytes_per_second * args.chunk_ms // 1000)

    async with websockets.connect(args.url, max_size=None) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "api_key": args.api_key,
                    "language": args.language,
                    "sample_rate": args.sample_rate,
                    "format": "pcm_s16le",
                },
                ensure_ascii=False,
            )
        )

        first = json.loads(await websocket.recv())
        print(json.dumps(first, ensure_ascii=False))
        if first.get("type") == "error":
            return

        sequence_tracker = SequenceTracker()
        if warning := sequence_tracker.observe(first):
            print(f"warning: {warning}", file=sys.stderr)

        receiver = asyncio.create_task(
            receive_messages(websocket, args.print_mode, sequence_tracker)
        )
        process = start_ffmpeg(args.audio_file, args.sample_rate)

        assert process.stdout is not None
        start = time.monotonic()
        sent_chunks = 0

        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            await websocket.send(chunk)
            sent_chunks += 1
            if args.realtime:
                expected_elapsed = sent_chunks * args.chunk_ms / 1000
                sleep_for = expected_elapsed - (time.monotonic() - start)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        process.wait()
        await websocket.send(json.dumps({"type": "end"}))
        await receiver


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(stream_audio(args))
    except FileNotFoundError as exc:
        if exc.filename == "ffmpeg":
            raise SystemExit("ffmpeg is required. Install it with: sudo apt install -y ffmpeg")
        raise


if __name__ == "__main__":
    main()
