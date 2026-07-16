import asyncio
from dataclasses import dataclass

import pytest

from app.asr import StreamingTranscriptionResult
from app.asr_gateway_backends import DispatchMode, ResultMode
from app.asr_gateway_local_adapter import LocalCoordinatorAdapter
from app.asr_gateway_scheduler import BatchKey, InferenceJob


@dataclass
class Snapshot:
    ready: bool = True
    accepting: bool = True
    active_streams: int = 0
    queue_depth: int = 0
    queued_audio_seconds: float = 0
    load_error: str | None = None


class FakeCoordinator:
    def __init__(self): self.calls = []; self.fail = False
    async def start(self): self.calls.append(("start",))
    async def stop(self): self.calls.append(("stop",))
    async def create_stream(self, language): self.calls.append(("open", language)); return "backend-1"
    async def add_audio(self, sid, pcm, rate):
        self.calls.append(("add", sid, len(pcm), rate))
        if self.fail: raise RuntimeError("credential secret")
        return StreamingTranscriptionResult(segment_id=1, segment_text="hello", decoded_samples_delta=len(pcm)//2)
    async def finish_segment(self, sid): self.calls.append(("segment", sid)); return StreamingTranscriptionResult("sentence", processed_samples=0, segment_finished=True)
    async def finish_stream(self, sid): self.calls.append(("finish", sid)); return StreamingTranscriptionResult("final", processed_samples=0, segment_finished=True)
    async def abort_stream(self, sid): self.calls.append(("abort", sid))
    def snapshot(self): return Snapshot()


def make_job(sid="public"):
    return InferenceJob("j", sid, 1, 1, "local", "backend-1", 0, 2, b"\x00\x00"*2, 1, BatchKey("local","rev","zh","transcribe",False,"","","pcm_s16le",0))


def test_adapter_lifecycle_serial_submit_and_snapshot():
    async def scenario():
        coordinator = FakeCoordinator()
        adapter = LocalCoordinatorAdapter(lambda: coordinator, worker_id="local", model_id="Qwen/Qwen3-ASR-1.7B", model_revision="rev", gpu_id="gpu-0")
        await adapter.warmup()
        backend = await adapter.open_session("public", language="zh")
        results = await adapter.submit([make_job()])
        segment = await adapter.finish_segment("public")
        final = await adapter.finish_session("public")
        await adapter.abort_session("public")
        snapshot = await adapter.snapshot()
        await adapter.close()
        return adapter, coordinator.calls, backend, results, segment, final, snapshot
    adapter, calls, backend, results, segment, final, snapshot = asyncio.run(scenario())
    assert adapter.capabilities.dispatch_mode is DispatchMode.SINGLE
    assert adapter.capabilities.max_batch_items == 1
    assert adapter.capabilities.result_mode is ResultMode.REPLACEABLE_SEGMENT
    assert backend == "backend-1"
    assert results[0].text == "hello" and results[0].end_sample == 2
    assert results[0].segment_id == 1
    assert segment.text == "sentence" and final.text == "final"
    assert calls[0] == ("start",) and calls[-1] == ("stop",)
    assert snapshot["ready"] is True


def test_adapter_rejects_batch_stale_and_sanitizes_failure():
    async def scenario():
        coordinator = FakeCoordinator()
        adapter = LocalCoordinatorAdapter(lambda: coordinator, worker_id="local", model_id="m", model_revision="r", gpu_id="g")
        await adapter.warmup(); await adapter.open_session("public")
        with pytest.raises(ValueError, match="one item"): await adapter.submit([make_job(), make_job("other")])
        coordinator.fail = True
        result = (await adapter.submit([make_job()]))[0]
        await adapter.abort_session("public")
        with pytest.raises(KeyError, match="stale session"): await adapter.submit([make_job()])
        await adapter.close()
        return result
    result = asyncio.run(scenario())
    assert result.error == "RuntimeError: backend operation failed"
    assert "secret" not in result.error
