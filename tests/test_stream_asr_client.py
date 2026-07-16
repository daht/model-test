import argparse
import asyncio
import json
from types import SimpleNamespace

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from scripts import stream_asr_client


def test_display_state_appends_sentence_final_and_replaces_partial():
    state = stream_asr_client.DisplayState()

    assert state.apply({"type": "partial", "text": "可以到店"}) == "可以到店"
    assert state.apply({"type": "partial", "text": "可以到店使用"}) == "可以到店使用"
    assert state.apply({"type": "sentence_final", "text": "可以到店"}) == "可以到店"
    assert state.apply({"type": "partial", "text": "使用也可以打包"}) == "可以到店使用也可以打包"


def test_display_state_final_uses_remaining_tail():
    state = stream_asr_client.DisplayState()
    state.apply({"type": "sentence_final", "text": "hello"})

    assert state.apply({"type": "final", "text": " world"}) == "hello world"


def test_empty_partial_after_commit_does_not_duplicate_display():
    state = stream_asr_client.DisplayState()

    assert state.apply({"type": "sentence_final", "text": "hello"}) == "hello"
    assert state.apply({"type": "partial", "text": ""}) == "hello"


def test_sequence_tracker_reports_gap_and_non_increasing_sequence():
    tracker = stream_asr_client.SequenceTracker()

    assert tracker.observe({"sequence": 1}) is None
    assert tracker.observe({"sequence": 3}) == "server event sequence gap: expected 2, got 3"
    assert tracker.observe({"sequence": 3}) == "server event sequence is not increasing: 3 after 3"


def test_strict_ready_validation_requires_ready_at_sequence_one():
    tracker = stream_asr_client.SequenceTracker()

    stream_asr_client.validate_ready_event(
        {"type": "ready", "sequence": 1},
        tracker,
        verify_protocol=True,
    )
    with pytest.raises(stream_asr_client.StreamClientError, match="ready"):
        stream_asr_client.validate_ready_event(
            {"type": "partial", "sequence": 1},
            stream_asr_client.SequenceTracker(),
            verify_protocol=True,
        )
    with pytest.raises(stream_asr_client.StreamClientError, match="sequence"):
        stream_asr_client.validate_ready_event(
            {"type": "ready", "sequence": 2},
            stream_asr_client.SequenceTracker(),
            verify_protocol=True,
        )


def test_strict_protocol_requires_continuous_sequence_and_sentence_final():
    class WebSocket:
        def __init__(self, payloads, *, close_code=1000, close_error=None):
            self.payloads = iter(json.dumps(payload) for payload in payloads)
            self.protocol = SimpleNamespace(close_rcvd=Close(close_code, ""))
            self.close_error = close_error

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.payloads)
            except StopIteration:
                if self.close_error is not None:
                    error, self.close_error = self.close_error, None
                    raise error
                raise StopAsyncIteration from None

    async def receive(payloads, **websocket_options):
        tracker = stream_asr_client.SequenceTracker()
        tracker.observe({"type": "ready", "sequence": 1})
        await stream_asr_client.receive_messages(
            WebSocket(payloads, **websocket_options),
            sequence_tracker=tracker,
            verify_protocol=True,
        )

    asyncio.run(
        receive(
            [
                {"type": "sentence_final", "sequence": 2, "text": "hello"},
                {"type": "final", "sequence": 3, "text": ""},
            ]
        )
    )
    with pytest.raises(stream_asr_client.StreamClientError, match="sequence"):
        asyncio.run(
            receive(
                [
                    {"type": "sentence_final", "sequence": 3, "text": "hello"},
                    {"type": "final", "sequence": 4, "text": ""},
                ]
            )
        )
    with pytest.raises(stream_asr_client.StreamClientError, match="sequence"):
        asyncio.run(
            receive(
                [
                    {"type": "sentence_final", "text": "hello"},
                    {"type": "final", "sequence": 2, "text": ""},
                ]
            )
        )
    with pytest.raises(stream_asr_client.StreamClientError, match="sentence_final"):
        asyncio.run(receive([{"type": "final", "sequence": 2, "text": ""}]))


def test_strict_protocol_rejects_abnormal_close_after_final():
    class WebSocket:
        def __init__(self):
            self.protocol = SimpleNamespace(close_rcvd=Close(1011, "server failure"))
            self.payloads = iter(
                json.dumps(payload)
                for payload in (
                    {"type": "sentence_final", "sequence": 2, "text": "speech"},
                    {"type": "final", "sequence": 3, "text": ""},
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.payloads)
            except StopIteration:
                raise ConnectionClosedError(
                    Close(1011, "server failure"),
                    None,
                    None,
                ) from None

    tracker = stream_asr_client.SequenceTracker()
    tracker.observe({"type": "ready", "sequence": 1})

    with pytest.raises(stream_asr_client.StreamClientError, match="close.*1011"):
        asyncio.run(
            stream_asr_client.receive_messages(
                WebSocket(),
                sequence_tracker=tracker,
                verify_protocol=True,
            )
        )


def test_strict_protocol_rejects_event_after_final():
    class WebSocket:
        def __init__(self):
            self.protocol = SimpleNamespace(close_rcvd=Close(1000, ""))
            self.payloads = iter(
                json.dumps(payload)
                for payload in (
                    {"type": "sentence_final", "sequence": 2, "text": "speech"},
                    {"type": "final", "sequence": 3, "text": ""},
                    {"type": "partial", "sequence": 4, "text": "late"},
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.payloads)
            except StopIteration:
                raise StopAsyncIteration from None

    tracker = stream_asr_client.SequenceTracker()
    tracker.observe({"type": "ready", "sequence": 1})

    with pytest.raises(stream_asr_client.StreamClientError, match="after final"):
        asyncio.run(
            stream_asr_client.receive_messages(
                WebSocket(),
                sequence_tracker=tracker,
                verify_protocol=True,
            )
        )


def test_strict_protocol_rejects_malformed_json_immediately(capsys):
    class WebSocket:
        protocol = SimpleNamespace(close_rcvd=Close(1000, ""))

        def __init__(self):
            self.messages = iter(
                (
                    "not-json",
                    json.dumps(
                        {"type": "sentence_final", "sequence": 2, "text": "speech"}
                    ),
                    json.dumps({"type": "final", "sequence": 3, "text": ""}),
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise StopAsyncIteration from None

    tracker = stream_asr_client.SequenceTracker()
    tracker.observe({"type": "ready", "sequence": 1})

    with pytest.raises(stream_asr_client.StreamClientError, match="malformed JSON"):
        asyncio.run(
            stream_asr_client.receive_messages(
                WebSocket(),
                sequence_tracker=tracker,
                verify_protocol=True,
            )
        )
    assert "not-json" not in capsys.readouterr().out


def test_non_strict_protocol_keeps_malformed_json_visible(capsys):
    class WebSocket:
        close_code = 1000

        def __init__(self):
            self.messages = iter(("not-json", json.dumps({"type": "final"})))

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise StopAsyncIteration from None

    asyncio.run(stream_asr_client.receive_messages(WebSocket()))

    assert "not-json" in capsys.readouterr().out


def test_strict_protocol_rejects_sent_only_normal_close():
    class WebSocket:
        protocol = SimpleNamespace(
            close_rcvd=Close(1000, "contradictory protocol state")
        )

        def __init__(self):
            self.messages = iter(
                json.dumps(payload)
                for payload in (
                    {"type": "sentence_final", "sequence": 2, "text": "speech"},
                    {"type": "final", "sequence": 3, "text": ""},
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise ConnectionClosedError(
                    None,
                    Close(1000, "local close"),
                    None,
                ) from None

    tracker = stream_asr_client.SequenceTracker()
    tracker.observe({"type": "ready", "sequence": 1})

    with pytest.raises(stream_asr_client.StreamClientError, match="received.*close"):
        asyncio.run(
            stream_asr_client.receive_messages(
                WebSocket(),
                sequence_tracker=tracker,
                verify_protocol=True,
            )
        )


def test_strict_protocol_accepts_received_normal_close_after_final():
    class WebSocket:
        protocol = SimpleNamespace(close_rcvd=Close(1000, "normal"))

        def __init__(self):
            self.messages = iter(
                json.dumps(payload)
                for payload in (
                    {"type": "sentence_final", "sequence": 2, "text": "speech"},
                    {"type": "final", "sequence": 3, "text": ""},
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise ConnectionClosedOK(
                    Close(1000, "normal"),
                    Close(1000, "normal"),
                    True,
                ) from None

    tracker = stream_asr_client.SequenceTracker()
    tracker.observe({"type": "ready", "sequence": 1})

    asyncio.run(
        stream_asr_client.receive_messages(
            WebSocket(),
            sequence_tracker=tracker,
            verify_protocol=True,
        )
    )


def test_api_key_source_is_tracked_and_strict_mode_requires_environment(monkeypatch):
    environment_marker = "environment-source-marker"
    argument_marker = "argument-source-marker"
    monkeypatch.setenv("API_KEY", environment_marker)

    environment_args = stream_asr_client.parse_args(
        ["external-audio.flac", "--verify-protocol"]
    )
    argument_args = stream_asr_client.parse_args(
        [
            "external-audio.flac",
            "--verify-protocol",
            "--api-key",
            argument_marker,
        ]
    )
    manual_args = stream_asr_client.parse_args(
        ["external-audio.flac", "--api-key", argument_marker]
    )

    assert environment_args.api_key_source == "environment"
    assert argument_args.api_key_source == "argument"
    assert manual_args.api_key_source == "argument"
    stream_asr_client.validate_api_key_input(environment_args)
    stream_asr_client.validate_api_key_input(manual_args)
    with pytest.raises(SystemExit, match="API_KEY environment"):
        stream_asr_client.validate_api_key_input(argument_args)


def test_strict_mode_missing_key_names_only_environment_input(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    args = stream_asr_client.parse_args(["external-audio.flac", "--verify-protocol"])

    assert args.api_key_source == "missing"
    with pytest.raises(SystemExit, match="Missing API_KEY environment variable"):
        stream_asr_client.validate_api_key_input(args)


def test_stream_runtime_rejects_strict_argument_key_without_logging_it(
    monkeypatch, capsys
):
    argument_marker = "argument-source-marker"
    monkeypatch.delenv("API_KEY", raising=False)
    args = stream_asr_client.parse_args(
        [
            "external-audio.flac",
            "--verify-protocol",
            "--api-key",
            argument_marker,
        ]
    )

    with pytest.raises(SystemExit, match="API_KEY environment"):
        asyncio.run(stream_asr_client.stream_audio(args))

    captured = capsys.readouterr()
    assert argument_marker not in captured.out
    assert argument_marker not in captured.err


def test_cli_help_explains_strict_credential_source(capsys):
    with pytest.raises(SystemExit) as exc_info:
        stream_asr_client.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "manual non-strict use" in help_text
    assert "API_KEY environment variable" in help_text


def test_legacy_stream_info_selects_payload_key_and_end_command():
    stream_info = {
        "protocol_version": 2,
        "start_message": {
            "type": "start",
            "api_key": "<your-api-key>",
            "language": "zh",
        },
        "end_message": {"type": "end"},
    }

    assert stream_asr_client.detect_protocol(stream_info) == "legacy"
    assert stream_asr_client.build_start_message(
        protocol="legacy",
        api_key="runtime-key",
        language="zh",
        sample_rate=16000,
    ) == {
        "type": "start",
        "api_key": "runtime-key",
        "language": "zh",
        "sample_rate": 16000,
        "format": "pcm_s16le",
    }
    assert stream_asr_client.finish_message("legacy") == {"type": "end"}


def test_gateway_stream_info_selects_header_auth_and_finish_command():
    stream_info = {
        "protocol_version": 2,
        "websocket_url": "/v1/transcribe/stream",
        "format": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
    }

    assert stream_asr_client.detect_protocol(stream_info) == "gateway"
    assert stream_asr_client.build_start_message(
        protocol="gateway",
        api_key="runtime-key",
        language="zh",
        sample_rate=16000,
    ) == {
        "type": "start",
        "language": "zh",
        "sample_rate": 16000,
        "format": "pcm_s16le",
    }
    assert stream_asr_client.finish_message("gateway") == {"type": "finish"}


def test_error_payload_is_not_treated_as_transcript():
    state = stream_asr_client.DisplayState()

    assert state.apply({"type": "error", "code": "server_busy", "text": "secret"}) is None


def test_server_close_cancels_sender_and_reaps_ffmpeg_without_task_leak():
    class BlockingStdout:
        async def read(self, _size):
            await asyncio.Event().wait()

    class Process:
        def __init__(self):
            self.stdout = BlockingStdout()
            self.returncode = None
            self.terminated = False
            self.killed = False
            self._killed = asyncio.Event()

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.returncode = -9
            self._killed.set()

        async def wait(self):
            await self._killed.wait()
            return self.returncode

    class ClosedWebSocket:
        async def send(self, _payload):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def scenario():
        process = Process()
        args = argparse.Namespace(
            chunk_ms=200,
            realtime=False,
            print_mode="events",
        )
        with pytest.raises(stream_asr_client.StreamClientError, match="before final"):
            await asyncio.wait_for(
                stream_asr_client.run_stream_tasks(
                    ClosedWebSocket(),
                    process,
                    args,
                    chunk_size=6400,
                    sequence_tracker=stream_asr_client.SequenceTracker(),
                ),
                timeout=2.0,
            )
        return process

    process = asyncio.run(scenario())

    assert process.terminated is True
    assert process.killed is True


def test_ffmpeg_cleanup_failure_does_not_hide_primary_stream_error(monkeypatch):
    class BlockingStdout:
        async def read(self, _size):
            await asyncio.Event().wait()

    class Process:
        stdout = BlockingStdout()
        returncode = None

    class ClosedWebSocket:
        async def send(self, _payload):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def broken_cleanup(_process):
        raise stream_asr_client.StreamClientError("ffmpeg could not be reaped")

    async def scenario():
        monkeypatch.setattr(stream_asr_client, "cleanup_ffmpeg", broken_cleanup)
        args = argparse.Namespace(
            chunk_ms=200,
            realtime=False,
            print_mode="events",
        )
        with pytest.raises(
            stream_asr_client.StreamClientError,
            match="server closed before final",
        ):
            await stream_asr_client.run_stream_tasks(
                ClosedWebSocket(),
                Process(),
                args,
                chunk_size=6400,
                sequence_tracker=stream_asr_client.SequenceTracker(),
            )

    asyncio.run(scenario())


def test_cleanup_drains_ffmpeg_stdout_after_kill_before_waiting():
    class Stdout:
        def __init__(self, process):
            self.process = process

        async def read(self, _size):
            await self.process.killed.wait()
            self.process.drained.set()
            return b""

    class Process:
        def __init__(self):
            self.returncode = None
            self.killed = asyncio.Event()
            self.drained = asyncio.Event()
            self.stdout = Stdout(self)

        def terminate(self):
            return None

        def kill(self):
            self.returncode = -9
            self.killed.set()

        async def wait(self):
            await self.drained.wait()
            return self.returncode

    async def scenario():
        process = Process()
        await stream_asr_client.cleanup_ffmpeg(process, timeout=0.01)
        return process

    process = asyncio.run(scenario())
    assert process.killed.is_set()
    assert process.drained.is_set()


def test_business_error_exits_cleanly_without_traceback(monkeypatch, capsys):
    async def fail(_args):
        raise stream_asr_client.StreamClientError("server closed before final")

    monkeypatch.setattr(stream_asr_client, "parse_args", lambda: object())
    monkeypatch.setattr(stream_asr_client, "stream_audio", fail)

    with pytest.raises(SystemExit) as exc_info:
        stream_asr_client.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "ASR stream failed: server closed before final" in captured.err
    assert "Traceback" not in captured.err
