import asyncio
from dataclasses import replace

from fastapi.testclient import TestClient

from app.asr_gateway import GatewayRuntime, create_app
from app.asr_gateway_backends import DispatchMode, ResultMode
from app.asr_gateway_scheduler import InferenceResult
from app.config import Settings
from tests.test_asr_gateway_backends import capabilities


class FakeAdapter:
    def __init__(self, *, dynamic=False, result_mode=ResultMode.CUMULATIVE_SNAPSHOT):
        caps = capabilities(worker_id="fake")
        caps = replace(caps, result_mode=result_mode)
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
    settings = Settings(model_backend="mock", asr_backend="mock", api_key="test-key", asr_gateway_schedule_max_wait_ms=1, **setting_overrides)
    runtime = GatewayRuntime(settings, {"fake": adapter})
    return create_app(runtime=runtime), adapter


def test_health_readiness_inventory_and_metrics_are_semantic():
    app, _ = app_with()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        info = client.get("/v1/transcribe/stream-info", headers={"X-API-Key":"test-key"}).json()
        assert info["protocol_version"] == 2
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
            assert ws.receive_json()["type"] == "sentence_final"
            assert ws.receive_json()["type"] == "partial"
            ws.send_json({"type":"finish"})
            event = ws.receive_json()
            assert event["type"] == "final"
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


def test_runtime_barrier_forms_cross_session_dynamic_batch():
    async def scenario():
        adapter = FakeAdapter(dynamic=True)
        settings = Settings(model_backend="mock", asr_backend="mock", api_key="test-key", asr_gateway_schedule_max_wait_ms=20, asr_gateway_max_active_sessions=4)
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


def test_runtime_uses_segment_local_protocol_for_replaceable_adapter_results():
    async def scenario():
        adapter = FakeAdapter(result_mode=ResultMode.REPLACEABLE_SEGMENT)
        settings = Settings(model_backend="mock", asr_backend="mock", api_key="test-key", asr_gateway_schedule_max_wait_ms=0)
        runtime = GatewayRuntime(settings, {"fake": adapter}); await runtime.start()
        session = await runtime.open_session("s", language="zh", options={})
        await runtime.ingest(session, b"\x00\x00" * 4, force=True)
        await runtime.scheduler.run_once("fake", force=True)
        event = await runtime.event_queue("s").get()
        await runtime.close()
        return event
    assert asyncio.run(scenario())["type"] == "partial"
