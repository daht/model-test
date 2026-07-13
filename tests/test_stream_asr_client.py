import argparse
import asyncio
import json

import pytest
from websockets.exceptions import ConnectionClosedError
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
            self.close_code = close_code
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
        close_code = 1011

        def __init__(self):
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
        close_code = 1000

        def __init__(self):
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
