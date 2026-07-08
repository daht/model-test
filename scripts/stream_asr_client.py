#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import websockets


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
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between chunks to simulate microphone streaming.",
    )
    return parser.parse_args()


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


async def receive_messages(websocket: websockets.ClientConnection) -> None:
    async for message in websocket:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print(message)
            continue

        message_type = payload.get("type")
        if message_type in {"partial", "final"}:
            print(f"[{message_type}] {payload.get('text', '')}")
        else:
            print(json.dumps(payload, ensure_ascii=False))


async def stream_audio(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set API_KEY.")

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

        receiver = asyncio.create_task(receive_messages(websocket))
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
