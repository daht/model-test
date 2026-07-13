#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request

import websockets


class StreamClientError(RuntimeError):
    pass


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
    parser.add_argument(
        "--verify-protocol",
        action="store_true",
        help="Require strict sequences, sentence_final, one terminal final, and close code 1000.",
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


async def start_ffmpeg(
    audio_file: str, sample_rate: int
) -> asyncio.subprocess.Process:
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
    return await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def receive_messages(
    websocket: websockets.ClientConnection,
    print_mode: str = "events",
    sequence_tracker: SequenceTracker | None = None,
    verify_protocol: bool = False,
) -> None:
    display_state = DisplayState()
    tracker = sequence_tracker or SequenceTracker()
    final_count = 0
    sentence_final_count = 0
    close_code = None
    try:
        async for message in websocket:
            if verify_protocol and final_count:
                raise StreamClientError("server sent an event after final")
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(message)
                continue
            if not isinstance(payload, dict):
                raise StreamClientError("server returned a non-object event")
            sequence = payload.get("sequence")
            if verify_protocol and (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence <= 0
            ):
                raise StreamClientError(
                    f"event sequence must be a positive integer, got {sequence!r}"
                )
            if warning := tracker.observe(payload):
                if verify_protocol:
                    raise StreamClientError(warning)
                print(f"warning: {warning}", file=sys.stderr)

            message_type = payload.get("type")
            if message_type == "error":
                raise StreamClientError(
                    f"server error {payload.get('code', 'unknown')}: "
                    f"{payload.get('message', 'ASR stream failed')}"
                )
            if message_type == "final":
                final_count += 1
            elif message_type == "sentence_final":
                sentence_final_count += 1
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
    except websockets.ConnectionClosed as exc:
        if exc.rcvd is not None:
            close_code = exc.rcvd.code
        elif exc.sent is not None:
            close_code = exc.sent.code
        else:
            close_code = None
        if verify_protocol and close_code != 1000:
            raise StreamClientError(
                f"server closed with code {close_code}; expected normal close code 1000"
            ) from exc
        if final_count != 1:
            raise StreamClientError("server closed before final") from exc
    if close_code is None:
        close_code = getattr(websocket, "close_code", None)
    if verify_protocol and close_code != 1000:
        raise StreamClientError(
            f"server closed with code {close_code!r}; expected normal close code 1000"
        )
    if final_count != 1:
        if final_count == 0:
            raise StreamClientError("server closed before final")
        raise StreamClientError(f"server sent {final_count} final events")
    if verify_protocol and sentence_final_count == 0:
        raise StreamClientError("server sent no sentence_final event for speech audio")


def validate_ready_event(
    payload: dict[str, object],
    sequence_tracker: SequenceTracker,
    *,
    verify_protocol: bool,
) -> str | None:
    if payload.get("type") != "ready":
        raise StreamClientError(f"expected ready event, got {payload!r}")
    sequence = payload.get("sequence")
    if verify_protocol and (
        isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != 1
    ):
        raise StreamClientError(
            f"ready event sequence must start at 1, got {sequence!r}"
        )
    return sequence_tracker.observe(payload)


async def send_audio(
    websocket,
    process: asyncio.subprocess.Process,
    args: argparse.Namespace,
    chunk_size: int,
) -> None:
    assert process.stdout is not None
    start = time.monotonic()
    sent_chunks = 0
    while True:
        chunk = await process.stdout.read(chunk_size)
        if not chunk:
            break
        try:
            await websocket.send(chunk)
        except websockets.ConnectionClosed as exc:
            raise StreamClientError("server closed while audio was being sent") from exc
        sent_chunks += 1
        if args.realtime:
            expected_elapsed = sent_chunks * args.chunk_ms / 1000
            sleep_for = expected_elapsed - (time.monotonic() - start)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    return_code = await process.wait()
    if return_code:
        raise StreamClientError(f"ffmpeg exited with status {return_code}")
    try:
        await websocket.send(json.dumps({"type": "end"}))
    except websockets.ConnectionClosed as exc:
        raise StreamClientError("server closed before end could be sent") from exc


async def cleanup_ffmpeg(
    process: asyncio.subprocess.Process, *, timeout: float = 0.75
) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            await process.wait()
            return
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError as exc:
        raise StreamClientError("ffmpeg could not be reaped") from exc


async def run_stream_tasks(
    websocket,
    process: asyncio.subprocess.Process,
    args: argparse.Namespace,
    *,
    chunk_size: int,
    sequence_tracker: SequenceTracker,
) -> None:
    sender = asyncio.create_task(send_audio(websocket, process, args, chunk_size))
    receiver = asyncio.create_task(
        receive_messages(
            websocket,
            args.print_mode,
            sequence_tracker,
            verify_protocol=getattr(args, "verify_protocol", False),
        )
    )
    tasks = {sender, receiver}
    try:
        done, _pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            exception = task.exception()
            if exception is not None:
                raise exception
        if receiver in done and not sender.done():
            raise StreamClientError("server closed before final audio was sent")
        if sender in done:
            await receiver
        else:
            await sender
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await cleanup_ffmpeg(process)


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

        try:
            first = json.loads(await websocket.recv())
        except (json.JSONDecodeError, TypeError) as exc:
            raise StreamClientError("server returned an invalid ready response") from exc
        if not isinstance(first, dict):
            raise StreamClientError("server returned a non-object ready response")
        print(json.dumps(first, ensure_ascii=False))
        if first.get("type") == "error":
            raise StreamClientError(
                f"server error {first.get('code', 'unknown')}: "
                f"{first.get('message', 'ASR stream failed')}"
            )

        sequence_tracker = SequenceTracker()
        if warning := validate_ready_event(
            first,
            sequence_tracker,
            verify_protocol=getattr(args, "verify_protocol", False),
        ):
            print(f"warning: {warning}", file=sys.stderr)

        process = await start_ffmpeg(args.audio_file, args.sample_rate)
        await run_stream_tasks(
            websocket,
            process,
            args,
            chunk_size=chunk_size,
            sequence_tracker=sequence_tracker,
        )


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(stream_audio(args))
    except FileNotFoundError as exc:
        if exc.filename == "ffmpeg":
            raise SystemExit("ffmpeg is required. Install it with: sudo apt install -y ffmpeg")
        raise
    except StreamClientError as exc:
        print(f"ASR stream failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (OSError, websockets.WebSocketException) as exc:
        print(f"ASR connection failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
