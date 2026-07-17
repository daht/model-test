import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.asr_gateway import GatewayRuntime, OutboundEvent, _default_runtime, create_app, send_outbound_events
from app.asr_gateway_backends import DispatchMode, ResultMode, StreamingMode, VadMode
from app.asr_gateway_scheduler import InferenceResult
from app.config import Settings
from scripts import stream_asr_client
from tests.test_asr_gateway_backends import capabilities


def read_env_example(path: str) -> dict[str, str]:
    values = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name] = value
    return values


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeAdapter:
    def __init__(self, *, dynamic=False, result_mode=ResultMode.CUMULATIVE_SNAPSHOT):
        caps = capabilities(worker_id="fake")
        caps = replace(caps, result_mode=result_mode, vad_mode=VadMode.NONE)
        self.capabilities = replace(caps, dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH, max_batch_items=4, session_capacity=4) if dynamic else caps
        self.sessions = set(); self.calls = []; self.started = False
    async def warmup(self): self.started = True
    async def open_session(self, session_id, **options): self.sessions.add(session_id); return f"b-{session_id}"
    async def submit(self, jobs):
        self.calls.append(list(jobs))
        return [InferenceResult.from_job(job, text=f"heard-{job.sample_count}") for job in jobs]
    async def finish_session(self, session_id): self.sessions.discard(session_id); return type("R", (), {"tail_text":"done", "text":"done"})()
    async def finish_segment(self, session_id): return type("R", (), {"tail_text":"segment", "text":"segment"})()
    async def abort_session(self, session_id): self.sessions.discard(session_id)
    async def close(self): self.started = False
    async def snapshot(self): return {"ready": self.started, "accepting": self.started}


def app_with(adapter=None, **setting_overrides):
    adapter = adapter or FakeAdapter()
    settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_schedule_max_wait_ms=1, asr_gateway_default_backend="fake", **setting_overrides)
    runtime = GatewayRuntime(settings, {"fake": adapter})
    return create_app(runtime=runtime), adapter


def test_health_readiness_inventory_and_metrics_are_semantic():
    app, _ = app_with()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        info = client.get("/v1/transcribe/stream-info", headers={"X-API-Key":"test-key"}).json()
        assert info["protocol_version"] == 2
        assert client.get("/v1/transcribe/stream-info").status_code == 401
        assert client.get("/v1/asr/backends").status_code == 401
        assert client.get("/v1/asr/backends", headers={"X-API-Key":"test-key"}).json()["ready"] is True
        assert "completed_jobs" in client.get("/v1/asr/metrics", headers={"X-API-Key":"test-key"}).json()


def test_websocket_auth_start_audio_segment_finish_and_close_1000():
    app, adapter = app_with(asr_gateway_default_update_ms=1)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/transcribe/stream", headers={"X-API-Key":"test-key"}) as ws:
            ws.send_json({"type":"start", "format":"pcm_s16le", "sample_rate":16000, "channels":1, "language":"zh"})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(b"\x00\x00" * 16)
            assert ws.receive_json()["type"] == "partial"
            ws.send_json({"type":"segment"})
            segment_events = [ws.receive_json() for _ in range(3)]
            assert [event["type"] for event in segment_events] == [
                "partial", "sentence_final", "partial"
            ]
            ws.send_json({"type":"finish"})
            finish_events = [ws.receive_json() for _ in range(2)]
            assert [event["type"] for event in finish_events] == ["partial", "final"]
            assert [event["sequence"] for event in segment_events + finish_events] == [3, 4, 5, 6, 7]
        assert adapter.sessions == set()


def test_invalid_start_and_odd_audio_fail_explicitly_once():
    app, _ = app_with()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/transcribe/stream", headers={"X-API-Key":"test-key"}) as ws:
            ws.send_json({"type":"start", "format":"wav", "sample_rate":16000, "channels":1})
            event = ws.receive_json()
            assert event["type"] == "error" and event["sequence"] == 1
        with client.websocket_connect("/v1/transcribe/stream", headers={"X-API-Key":"test-key"}) as ws:
            ws.send_json({"type":"start", "format":"pcm_s16le", "sample_rate":16000, "channels":1})
            ws.receive_json(); ws.send_bytes(b"x")
            assert ws.receive_json()["type"] == "error"


def test_unauthorized_and_overload_fail_before_admission():
    app, _ = app_with(asr_gateway_max_active_sessions=1)
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/v1/transcribe/stream"):
                raise AssertionError("unauthorized websocket was accepted")
        except Exception:
            pass
        with client.websocket_connect("/v1/transcribe/stream", headers={"X-API-Key":"test-key"}) as first:
            first.send_json({"type":"start"}); first.receive_json()
            try:
                with client.websocket_connect("/v1/transcribe/stream", headers={"X-API-Key":"test-key"}) as second:
                    second.send_json({"type":"start"})
                    assert second.receive_json()["code"] == "overloaded"
            except Exception:
                pass


def test_body_key_and_end_command_are_not_legacy_compatibility_paths():
    app, _ = app_with()
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/v1/transcribe/stream") as ws:
                ws.send_json({"type":"start", "api_key":"test-key"})
        with client.websocket_connect("/v1/transcribe/stream", headers={"X-API-Key":"test-key"}) as ws:
            ws.send_json({"type":"start"}); assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type":"end"})
            event = ws.receive_json()
            assert event["type"] == "error" and event["code"] == "invalid_command"


def test_runtime_barrier_forms_cross_session_dynamic_batch():
    async def scenario():
        adapter = FakeAdapter(dynamic=True)
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_schedule_max_wait_ms=20, asr_gateway_max_active_sessions=4, asr_gateway_default_backend="fake")
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        sessions = [await runtime.open_session(f"s{i}", language="zh", options={}) for i in range(3)]
        barrier = asyncio.Event()
        async def ingest(session): await barrier.wait(); await runtime.ingest(session, b"\x00\x00" * 100, force=True)
        tasks = [asyncio.create_task(ingest(s)) for s in sessions]
        barrier.set(); await asyncio.gather(*tasks)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.close()
        return [len(call) for call in adapter.calls]
    assert asyncio.run(scenario()) == [3]


def test_faster_whisper_candidate_coalesces_offset_sessions_before_buffer_pressure():
    async def scenario():
        values = read_env_example("cloud/A10.faster-whisper.env.example")
        clock = FakeClock()
        adapter = FakeAdapter(dynamic=True)
        adapter.capabilities = replace(
            adapter.capabilities,
            streaming_mode=StreamingMode.ROLLING,
        )
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="rolling",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_max_active_sessions=2,
            asr_gateway_schedule_max_wait_ms=int(
                values["ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS"]
            ),
            asr_gateway_max_session_buffer_seconds=float(
                values["ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS"]
            ),
            asr_gateway_max_queued_audio_seconds=16,
            asr_gateway_default_update_ms=2000,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter}, clock=clock)
        await adapter.warmup()
        await runtime.registry.register(adapter.capabilities)
        await runtime.registry.mark_ready("fake", True)
        first = await runtime.open_session("first", language="zh", options={})
        second = await runtime.open_session("second", language="zh", options={})
        frame = b"\x01\x00" * 32_000

        for _ in range(3):
            await runtime.ingest(first, frame, force=True)
            assert await runtime.scheduler.run_once("fake") == []
            clock.advance(0.1)
            assert await runtime.scheduler.run_once("fake") == []
            await runtime.ingest(second, frame, force=True)
            assert await runtime.scheduler.run_once("fake") == []
            clock.advance(0.101)
            results = await runtime.scheduler.run_once("fake")
            assert len(results) == 2

        snapshots = [first.sample_accounting, second.sample_accounting]
        await runtime.close()
        return [len(call) for call in adapter.calls], snapshots

    calls, snapshots = asyncio.run(scenario())

    assert calls == [2, 2, 2]
    assert all(
        item["buffered"] == item["reserved"] == 0 for item in snapshots
    )


def test_runtime_uses_segment_local_protocol_for_replaceable_adapter_results():
    async def scenario():
        adapter = FakeAdapter(result_mode=ResultMode.REPLACEABLE_SEGMENT)
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_schedule_max_wait_ms=0, asr_gateway_default_backend="fake")
        runtime = GatewayRuntime(settings, {"fake": adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        event = await runtime.event_queue("s").get()
        await runtime.close()
        return event
    assert asyncio.run(scenario()).payload["type"] == "partial"


def test_runtime_rejects_unsupported_language_before_backend_session_open():
    async def scenario():
        adapter = FakeAdapter()
        adapter.capabilities = replace(adapter.capabilities, languages=("zh",), backend_id="fake")
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake")
        runtime = GatewayRuntime(settings, {"fake": adapter}); await runtime.start()
        try:
            with pytest.raises(RuntimeError, match="unsupported backend route"):
                await runtime.open_session("ja", language="ja", options={})
        finally:
            await runtime.close()
        return adapter.sessions
    assert asyncio.run(scenario()) == set()


def test_runtime_rejects_duplicate_gpu_before_second_adapter_warmup():
    async def scenario():
        first = FakeAdapter(); second = FakeAdapter()
        first.capabilities = replace(first.capabilities, worker_id="first", backend_id="first", gpu_id="gpu-0")
        second.capabilities = replace(second.capabilities, worker_id="second", backend_id="second", gpu_id="gpu-0")
        first.warm_calls = second.warm_calls = 0
        async def warm_first(): first.warm_calls += 1
        async def warm_second(): second.warm_calls += 1
        first.warmup = warm_first; second.warmup = warm_second
        settings = Settings(model_backend="mock", asr_backend="mock", api_key="test-key")
        runtime = GatewayRuntime(settings, {"first": first, "second": second})
        with pytest.raises(ValueError, match="gpu_id"):
            await runtime.start()
        return first.warm_calls, second.warm_calls
    assert asyncio.run(scenario()) == (0, 0)


class FakeVad:
    def __init__(self): self.pending = bytearray(); self.finalized = 0
    def add_audio(self, pcm):
        self.pending.extend(pcm)
        if len(self.pending) < 8:
            return type("D", (), {"audio_to_model":b"", "endpoint":False, "discarded_samples":0})()
        audio = bytes(self.pending[:4]); del self.pending[:4]
        return type("D", (), {"audio_to_model":audio, "endpoint":True, "discarded_samples":0})()
    def endpoint_finalized(self):
        self.finalized += 1
        audio = bytes(self.pending); self.pending.clear()
        return type("D", (), {"audio_to_model":audio, "endpoint":False, "discarded_samples":0})()
    def finish_input(self):
        audio = bytes(self.pending); self.pending.clear()
        return type("D", (), {"audio_to_model":audio, "endpoint":False, "discarded_samples":0})()
    def reset(self): self.pending.clear()


def test_gateway_owned_vad_conserves_input_and_commits_endpoint():
    async def scenario():
        adapter = FakeAdapter(); adapter.capabilities = replace(adapter.capabilities, vad_mode=VadMode.GATEWAY)
        vad = FakeVad()
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake":adapter}, vad_factory=lambda: vad); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x01\x00" * 4)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.wait_idle("s")
        accounting = session.sample_accounting
        await runtime.abort("s"); await runtime.close()
        return accounting, vad.finalized, [call[0].sample_count for call in adapter.calls]
    accounting, finalized, submitted = asyncio.run(scenario())
    assert accounting["accepted"] == 4
    assert sum(accounting[key] for key in ("buffered","reserved","acknowledged","discarded","pending_vad")) == 4
    assert finalized == 1
    assert submitted == [2, 2]


def test_finish_unblocks_after_vad_discards_last_pending_sample():
    class FinishVad:
        def add_audio(self, pcm):
            return type(
                "D",
                (),
                {
                    "audio_to_model": pcm[:4],
                    "endpoint": False,
                    "discarded_samples": 0,
                },
            )()

        def finish_input(self):
            return type(
                "D",
                (),
                {
                    "audio_to_model": b"",
                    "endpoint": False,
                    "discarded_samples": 1,
                },
            )()

        def reset(self):
            return None

    async def scenario():
        adapter = FakeAdapter()
        adapter.capabilities = replace(adapter.capabilities, vad_mode=VadMode.GATEWAY)
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(
            settings, {"fake": adapter}, vad_factory=FinishVad
        )
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x01\x00" * 3, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        assert runtime._contexts["s"].idle.is_set() is False
        final = await asyncio.wait_for(runtime.finish("s"), timeout=0.1)
        accounting = session.sample_accounting
        await runtime.abort("s")
        await runtime.close()
        return final, accounting

    final, accounting = asyncio.run(scenario())

    assert final["type"] == "final"
    assert accounting["pending_vad"] == 0


def test_exact_maximum_rolls_backend_state_before_remainder_submit():
    async def scenario():
        adapter = FakeAdapter(); adapter.capabilities = replace(adapter.capabilities, preferred_chunk_samples=6, max_input_samples=6, max_segment_samples=6, max_batch_samples=6)
        order = []
        original_submit = adapter.submit
        async def submit(jobs): order.append(("submit", jobs[0].sample_count)); return await original_submit(jobs)
        async def segment(session_id): order.append(("rollover", session_id)); return type("R", (), {"text":"", "tail_text":""})()
        adapter.submit = submit; adapter.finish_segment = segment
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake":adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 12, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.abort("s"); await runtime.close()
        return order
    assert asyncio.run(scenario()) == [("submit",6), ("rollover","s"), ("submit",6), ("rollover","s")]


def test_runtime_submit_failure_clears_worker_readiness_and_fails_session():
    async def scenario():
        adapter = FakeAdapter()
        async def broken(jobs): raise RuntimeError("worker lost")
        adapter.submit = broken
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake":adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        event = await runtime.event_queue("s").get()
        snapshot = await runtime.registry.snapshot()
        backend_sessions = set(adapter.sessions)
        await runtime.abort("s"); await runtime.close()
        return event, snapshot, backend_sessions
    event, snapshot, backend_sessions = asyncio.run(scenario())
    assert event.payload["type"] == "error"
    assert snapshot["ready"] is False
    assert backend_sessions == set()


def test_runtime_records_all_job_timeline_stages_and_counters():
    async def scenario():
        adapter = FakeAdapter()
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake":adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        before = runtime.metrics.snapshot()["completed_jobs"]
        envelope = await runtime.event_queue("s").get()
        runtime.mark_event_sent(envelope.job_id)
        snapshot = runtime.metrics.snapshot()
        await runtime.abort("s")
        cancelled = runtime.metrics.snapshot()
        await runtime.close()
        return before, snapshot, cancelled
    before, snapshot, cancelled = asyncio.run(scenario())
    assert before == 0
    assert snapshot["completed_jobs"] == 1
    assert set(snapshot["latency"]) == {"chunk_wait_seconds", "batch_wait_seconds", "worker_wait_seconds", "inference_seconds", "egress_seconds"}
    assert cancelled["cancellations"] == 1


def test_runtime_buffer_gauges_track_held_reservation_and_cleanup():
    class HeldAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, jobs):
            self.calls.append(list(jobs))
            self.entered.set()
            await self.release.wait()
            return [
                InferenceResult.from_job(job, text=f"heard-{job.sample_count}")
                for job in jobs
            ]

    async def scenario():
        adapter = HeldAdapter()
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
            asr_gateway_default_update_ms=2000,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await adapter.warmup()
        await runtime.registry.register(adapter.capabilities)
        await runtime.registry.mark_ready("fake", True)
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x01\x00" * 32_000, force=True)
        submission = asyncio.create_task(
            runtime.scheduler.run_once("fake", force=True)
        )
        await adapter.entered.wait()
        held = runtime.metrics.snapshot()
        adapter.release.set()
        await submission
        cleared = runtime.metrics.snapshot()
        accounting = session.sample_accounting
        await runtime.close()
        return held, cleared, accounting

    held, cleared, accounting = asyncio.run(scenario())

    assert held["session_buffered_audio_seconds"] == 0
    assert held["session_reserved_audio_seconds"] == 2
    assert held["max_session_held_audio_seconds"] == 2
    assert held["session_buffer_high_water_seconds"] == 2
    assert cleared["session_buffered_audio_seconds"] == 0
    assert cleared["session_reserved_audio_seconds"] == 0
    assert cleared["max_session_held_audio_seconds"] == 0
    assert cleared["session_buffer_high_water_seconds"] == 2
    assert accounting["buffered"] == accounting["reserved"] == 0


def test_error_terminal_releases_lease_updates_gauges_and_logs(caplog):
    class WebSocket:
        def __init__(self):
            self.sent = []
            self.closed = []

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self, code):
            self.closed.append(code)

    async def scenario():
        adapter = FakeAdapter()
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        await runtime.open_session("s", language="zh", options={})
        active_before = runtime.metrics.snapshot()["active_sessions"]
        await runtime.enqueue_error(
            "s", BufferError("private details"), code="invalid_audio", close_code=1008
        )
        websocket = WebSocket()
        await send_outbound_events(runtime, runtime._contexts["s"], websocket)
        registry = await runtime.registry.snapshot()
        metrics = runtime.metrics.snapshot()
        await runtime.close()
        return active_before, websocket, registry, metrics

    with caplog.at_level("WARNING", logger="app.asr_gateway"):
        active_before, websocket, registry, metrics = asyncio.run(scenario())

    assert active_before == 1
    assert websocket.closed == [1008]
    assert websocket.sent[0]["type"] == "error"
    assert registry["workers"]["fake"]["active_leases"] == 0
    assert metrics["active_sessions"] == 0
    assert metrics["failures"] == 1
    assert "session_id=s code=invalid_audio exception_type=BufferError" in caplog.text
    assert "private details" not in caplog.text


def test_error_terminal_waits_for_accepted_job_and_aborts_backend_session():
    class WebSocket:
        async def send_json(self, payload):
            return None

        async def close(self, code):
            return None

    class HeldAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, jobs):
            self.calls.append(list(jobs))
            self.entered.set()
            await self.release.wait()
            return [
                InferenceResult.from_job(job, text=f"heard-{job.sample_count}")
                for job in jobs
            ]

    async def scenario():
        adapter = HeldAdapter()
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await adapter.warmup()
        await runtime.registry.register(adapter.capabilities)
        await runtime.registry.mark_ready("fake", True)
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x01\x00" * 4, force=True)
        dispatch = asyncio.create_task(
            runtime.scheduler.run_once("fake", force=True)
        )
        await adapter.entered.wait()

        failure = asyncio.create_task(
            runtime.enqueue_error("s", BufferError("private details"))
        )
        done, _ = await asyncio.wait({failure}, timeout=0)
        assert failure not in done

        adapter.release.set()
        await dispatch
        await failure
        ctx = runtime._contexts["s"]
        await send_outbound_events(runtime, ctx, WebSocket())
        metrics = runtime.metrics.snapshot()
        backend_sessions = set(adapter.sessions)
        contexts = set(runtime._contexts)
        await runtime.close()
        return metrics, backend_sessions, contexts

    metrics, backend_sessions, contexts = asyncio.run(scenario())

    assert metrics["failures"] == 1
    assert metrics["conflicts"] == 0
    assert backend_sessions == set()
    assert contexts == set()


def test_runtime_enforces_audio_session_and_undecoded_deadlines_with_fake_clock():
    async def scenario():
        clock = type("Clock", (), {"value":0.0, "__call__":lambda self:self.value})()
        adapter = FakeAdapter()
        settings = Settings(
            model_backend="mock", asr_backend="mock", asr_stream_mode="stateful",
            api_key="test-key", asr_gateway_default_backend="fake",
            asr_max_audio_seconds=.001, asr_max_session_seconds=.01,
            asr_max_undecoded_age_seconds=.005,
        )
        runtime = GatewayRuntime(settings, {"fake":adapter}, clock=clock); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        with pytest.raises(RuntimeError, match="audio duration"):
            await runtime.ingest(session, b"\x00\x00" * 17)
        await runtime.ingest(session, b"\x00\x00" * 8)
        clock.value = .006
        with pytest.raises(TimeoutError, match="undecoded"):
            await runtime.ingest(session, b"\x00\x00" * 1)
        clock.value = .011
        with pytest.raises(TimeoutError, match="session deadline"):
            runtime.check_deadlines("s")
        await runtime.abort("s"); await runtime.close()
    asyncio.run(scenario())


def test_continuous_successful_decode_progress_refreshes_undecoded_age():
    async def scenario():
        clock = type(
            "Clock", (), {"value": 0.0, "__call__": lambda self: self.value}
        )()
        adapter = FakeAdapter()
        adapter.capabilities = replace(
            adapter.capabilities,
            preferred_chunk_samples=4,
            max_input_samples=4,
            max_batch_samples=4,
        )
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
            asr_max_undecoded_age_seconds=0.005,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter}, clock=clock)
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 12, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        clock.value = 0.004
        await runtime.scheduler.run_once("fake", force=True)
        clock.value = 0.008
        await runtime.scheduler.run_once("fake", force=True)
        events = []
        queue = runtime.event_queue("s")
        while not queue.empty():
            events.append(queue.get_nowait().payload)
        await runtime.abort("s")
        await runtime.close()
        return [event["type"] for event in events], len(adapter.calls)

    assert asyncio.run(scenario()) == (["partial"], 3)


def test_drain_stops_new_routes_and_waits_for_sticky_session_release():
    async def scenario():
        adapter = FakeAdapter()
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_drain_timeout_seconds=1)
        runtime = GatewayRuntime(settings, {"fake":adapter}); await runtime.start()
        await runtime.open_session("s", language="zh", options={})
        draining = asyncio.create_task(runtime.drain_worker("fake"))
        await asyncio.wait({draining}, timeout=0)
        with pytest.raises(RuntimeError, match="unsupported backend route"):
            await runtime.open_session("new", language="zh", options={})
        assert draining.done() is False
        await runtime.abort("s"); await draining
        snapshot = await runtime.registry.snapshot()
        await runtime.close()
        return snapshot
    assert asyncio.run(scenario())["workers"] == {}


def test_single_outbound_owner_preserves_continuous_terminal_sequence():
    async def scenario():
        adapter = FakeAdapter()
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake": adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.enqueue_ready("s")
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.enqueue_segment("s")
        await runtime.enqueue_finish("s")
        envelopes = []
        while not runtime.event_queue("s").empty(): envelopes.append(await runtime.event_queue("s").get())
        await runtime.abort("s"); await runtime.close()
        return envelopes
    envelopes = asyncio.run(scenario())
    payloads = [envelope.payload for envelope in envelopes]
    tracker = stream_asr_client.SequenceTracker()
    for expected, payload in enumerate(payloads, start=1):
        assert payload["sequence"] == expected
        assert tracker.observe(payload) is None
    assert payloads[-1]["type"] == "final"
    assert envelopes[-1].terminal is True


def test_sender_backpressure_keeps_partial_before_final_and_metrics_uncompleted():
    class WebSocket:
        def __init__(self):
            self.entered = asyncio.Event(); self.release = asyncio.Event(); self.sent = []; self.closed = []
        async def send_json(self, payload):
            self.sent.append(payload); self.entered.set(); await self.release.wait()
        async def close(self, code): self.closed.append(code)

    async def scenario():
        adapter = FakeAdapter()
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake": adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        ctx = runtime._contexts["s"]
        async with ctx.protocol_lock:
            final = ctx.protocol.final()
            await ctx.events.put(OutboundEvent(final, terminal=True, close_code=1000))
        websocket = WebSocket()
        sender = asyncio.create_task(send_outbound_events(runtime, ctx, websocket))
        await websocket.entered.wait()
        before = runtime.metrics.snapshot()["completed_jobs"]
        websocket.release.set(); await sender
        after = runtime.metrics.snapshot()["completed_jobs"]
        await runtime.abort("s"); await runtime.close()
        return websocket.sent, websocket.closed, before, after
    sent, closed, before, after = asyncio.run(scenario())
    assert [event["type"] for event in sent] == ["partial", "final"]
    assert closed == [1000]
    assert before == 0 and after == 1


def test_sender_failure_does_not_claim_event_sent_or_complete_job():
    class BrokenWebSocket:
        async def send_json(self, payload): raise RuntimeError("disconnected")
        async def close(self, code): raise AssertionError("close must not follow failed send")

    async def scenario():
        adapter = FakeAdapter()
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake": adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        with pytest.raises(RuntimeError, match="disconnected"):
            await send_outbound_events(runtime, runtime._contexts["s"], BrokenWebSocket())
        completed = runtime.metrics.snapshot()["completed_jobs"]
        await runtime.abort("s"); await runtime.close()
        return completed
    assert asyncio.run(scenario()) == 0


def test_default_local_runtime_caps_jobs_to_stateful_frame_contract(monkeypatch):
    settings = Settings(
        model_backend="mock", asr_backend="mock", asr_stream_mode="stateful",
        api_key="test-key", asr_stream_chunk_seconds=2.0,
        asr_max_utterance_seconds=20.0, asr_max_frame_bytes=16_000,
    )
    monkeypatch.setattr("app.asr_gateway.get_settings", lambda: settings)
    runtime = _default_runtime()
    caps = runtime.adapters["local"].capabilities
    assert caps.preferred_chunk_samples == 8_000
    assert caps.max_input_samples == 8_000
    assert caps.max_segment_samples == 320_000
    assert caps.max_input_samples * 2 <= settings.asr_max_frame_bytes


def test_default_runtime_selects_faster_whisper_without_removing_qwen_rollback(monkeypatch):
    from app.asr_faster_whisper import FasterWhisperAdapter
    from app.asr_gateway_local_adapter import LocalCoordinatorAdapter

    faster_settings = Settings(
        _env_file=None,
        model_backend="mock",
        asr_backend="faster_whisper",
        asr_stream_mode="rolling",
        asr_model_name="large-v3",
        asr_model_id="/models/faster-whisper-large-v3",
        api_key="unit-test-only-not-a-production-secret-000000",
        asr_faster_whisper_batch_size=4,
    )
    monkeypatch.setattr("app.asr_gateway.get_settings", lambda: faster_settings)
    faster_runtime = _default_runtime()

    assert isinstance(faster_runtime.adapters["local"], FasterWhisperAdapter)
    assert faster_runtime.adapters["local"].capabilities.max_batch_items == 4

    qwen_settings = Settings(
        _env_file=None,
        model_backend="mock",
        asr_backend="qwen_vllm",
        asr_stream_mode="stateful",
        asr_model_id="Qwen/Qwen3-ASR-1.7B",
        api_key="unit-test-only-not-a-production-secret-000000",
    )
    monkeypatch.setattr("app.asr_gateway.get_settings", lambda: qwen_settings)
    qwen_runtime = _default_runtime()

    assert isinstance(qwen_runtime.adapters["local"], LocalCoordinatorAdapter)


def test_rolling_route_and_endpoint_jobs_use_final_decode_batch_identity():
    async def scenario():
        adapter = FakeAdapter(dynamic=True, result_mode=ResultMode.REPLACEABLE_SEGMENT)
        adapter.capabilities = replace(
            adapter.capabilities,
            streaming_mode=StreamingMode.ROLLING,
            vad_mode=VadMode.NONE,
            preferred_chunk_samples=4,
            max_input_samples=16,
            max_segment_samples=16,
            max_batch_samples=64,
        )
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="rolling",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        runtime._contexts["s"].endpoint_pending = True
        await runtime.ingest(session, b"\x01\x00" * 4, force=True)
        queued = runtime.scheduler._queues["fake"][0]
        await runtime.abort("s")
        await runtime.close()
        return runtime._required_streaming_mode(), queued

    mode, queued = asyncio.run(scenario())

    assert mode is StreamingMode.ROLLING
    assert queued.final is True
    assert queued.batch_key.decoding_identity.startswith("final:")


def test_faster_whisper_keeps_mixed_boundary_jobs_in_one_scheduler_batch():
    async def scenario():
        adapter = FakeAdapter(
            dynamic=True,
            result_mode=ResultMode.REPLACEABLE_SEGMENT,
        )
        adapter.capabilities = replace(
            adapter.capabilities,
            streaming_mode=StreamingMode.ROLLING,
            vad_mode=VadMode.NONE,
            preferred_chunk_samples=4,
            max_input_samples=16,
            max_segment_samples=16,
            max_batch_samples=64,
            model_id="/models/faster-whisper-large-v3",
        )
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="faster_whisper",
            asr_stream_mode="rolling",
            asr_model_name="large-v3",
            asr_model_id="/models/faster-whisper-large-v3",
            api_key="unit-test-only-not-a-production-secret-000000",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        final_session = await runtime.open_session("final", language="zh", options={})
        partial_session = await runtime.open_session("partial", language="zh", options={})
        runtime._contexts["final"].endpoint_pending = True
        await runtime.ingest(final_session, b"\x01\x00" * 4, force=True)
        await runtime.ingest(partial_session, b"\x02\x00" * 3, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        calls = [[(job.final, job.sample_count) for job in call] for call in adapter.calls]
        await runtime.close()
        return calls

    assert asyncio.run(scenario()) == [[(True, 4), (False, 3)]]


def test_exact_maximum_job_is_final_for_batched_beam_five_rollover():
    async def scenario():
        adapter = FakeAdapter(dynamic=True, result_mode=ResultMode.REPLACEABLE_SEGMENT)
        adapter.capabilities = replace(
            adapter.capabilities,
            preferred_chunk_samples=4,
            max_input_samples=4,
            max_segment_samples=4,
            max_batch_samples=16,
        )
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x01\x00" * 4, force=True)
        queued = runtime.scheduler._queues["fake"][0]
        await runtime.abort("s")
        await runtime.close()
        return queued

    queued = asyncio.run(scenario())

    assert queued.final is True
    assert queued.batch_key.decoding_identity.startswith("final:")


def test_backend_frame_jobs_do_not_roll_state_before_segment_boundary():
    async def scenario():
        adapter = FakeAdapter()
        adapter.capabilities = replace(
            adapter.capabilities,
            preferred_chunk_samples=4,
            max_input_samples=4,
            max_batch_samples=4,
            max_segment_samples=12,
        )
        rollovers = []

        async def finish_segment(session_id):
            rollovers.append(session_id)
            return type("R", (), {"text": "", "tail_text": ""})()

        adapter.finish_segment = finish_segment
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 8, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.abort("s")
        await runtime.close()
        return rollovers, [call[0].sample_count for call in adapter.calls]

    assert asyncio.run(scenario()) == ([], [4, 4])


def test_replaceable_backend_keeps_its_segment_identity_across_frame_jobs():
    async def scenario():
        adapter = FakeAdapter(result_mode=ResultMode.REPLACEABLE_SEGMENT)
        adapter.capabilities = replace(
            adapter.capabilities,
            preferred_chunk_samples=4,
            max_input_samples=4,
            max_batch_samples=4,
            max_segment_samples=12,
        )
        snapshots = iter(("first", "first second"))

        async def submit(jobs):
            adapter.calls.append(list(jobs))
            return [
                InferenceResult.from_job(
                    jobs[0], text=next(snapshots), segment_id=7
                )
            ]

        adapter.submit = submit
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 8, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        await runtime.scheduler.run_once("fake", force=True)
        events = []
        queue = runtime.event_queue("s")
        while not queue.empty():
            events.append(queue.get_nowait().payload)
        await runtime.abort("s")
        await runtime.close()
        return [(event["type"], event["text"]) for event in events]

    assert asyncio.run(scenario()) == [
        ("partial", "first"),
        ("partial", "first second"),
    ]


def test_twenty_second_boundary_rolls_state_before_one_sample_remainder():
    async def scenario():
        adapter = FakeAdapter(); adapter.capabilities = replace(adapter.capabilities, preferred_chunk_samples=32_000, max_input_samples=32_000, max_segment_samples=320_000, max_batch_samples=32_000)
        order = []; original = adapter.submit
        async def submit(jobs): order.append(("submit", jobs[0].sample_count)); return await original(jobs)
        async def segment(session_id): order.append(("rollover", session_id)); return type("R", (), {"text":"", "tail_text":""})()
        adapter.submit = submit; adapter.finish_segment = segment
        settings = Settings(model_backend="mock", asr_backend="mock", asr_stream_mode="stateful", api_key="test-key", asr_gateway_default_backend="fake", asr_gateway_schedule_max_wait_ms=0, asr_gateway_max_session_buffer_seconds=21, asr_gateway_max_queued_audio_seconds=22)
        runtime = GatewayRuntime(settings, {"fake":adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 320_001, force=True)
        for _ in range(11):
            await runtime.scheduler.run_once("fake", force=True)
        await runtime.abort("s"); await runtime.close()
        return order
    assert asyncio.run(scenario()) == [
        *(("submit", 32_000),) * 10,
        ("rollover", "s"),
        ("submit", 1),
    ]


def test_segment_boundary_splits_a_backend_frame_before_its_remainder():
    async def scenario():
        adapter = FakeAdapter()
        adapter.capabilities = replace(
            adapter.capabilities,
            preferred_chunk_samples=4,
            max_input_samples=4,
            max_segment_samples=10,
            max_batch_samples=4,
        )
        order = []
        original_submit = adapter.submit

        async def submit(jobs):
            order.append(("submit", jobs[0].sample_count))
            return await original_submit(jobs)

        async def segment(session_id):
            order.append(("rollover", session_id))
            return type("R", (), {"text": "", "tail_text": ""})()

        adapter.submit = submit
        adapter.finish_segment = segment
        settings = Settings(
            model_backend="mock",
            asr_backend="mock",
            asr_stream_mode="stateful",
            api_key="test-key",
            asr_gateway_default_backend="fake",
            asr_gateway_schedule_max_wait_ms=0,
        )
        runtime = GatewayRuntime(settings, {"fake": adapter})
        await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 12, force=True)
        for _ in range(4):
            await runtime.scheduler.run_once("fake", force=True)
        await runtime.abort("s")
        await runtime.close()
        return order

    assert asyncio.run(scenario()) == [
        ("submit", 4),
        ("submit", 4),
        ("submit", 2),
        ("rollover", "s"),
        ("submit", 2),
    ]
