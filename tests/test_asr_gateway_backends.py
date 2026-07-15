import asyncio

import pytest

from app.asr_gateway_backends import (
    BackendCapabilities,
    BackendRegistry,
    DispatchMode,
    ResultMode,
    StreamingMode,
    VadMode,
    WorkerLifecycle,
)


def capabilities(worker_id="worker-1", **overrides):
    values = dict(
        protocol_version=1,
        worker_id=worker_id,
        model_id="Qwen/Qwen3-ASR-1.7B",
        model_revision="rev-1",
        gpu_id="gpu-0",
        languages=("zh", "ja"),
        tasks=("transcribe",),
        streaming_mode=StreamingMode.STATEFUL,
        dispatch_mode=DispatchMode.SINGLE,
        vad_mode=VadMode.GATEWAY,
        result_mode=ResultMode.CUMULATIVE_SNAPSHOT,
        preferred_chunk_samples=24_000,
        max_input_samples=480_000,
        max_batch_items=1,
        max_batch_samples=480_000,
        max_in_flight=1,
        session_capacity=2,
        retry_safe=False,
        warmed=True,
    )
    values.update(overrides)
    return BackendCapabilities(**values)


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"worker_id": ""}, "worker_id"),
        ({"model_id": ""}, "model_id"),
        ({"max_in_flight": 0}, "max_in_flight"),
        ({"max_batch_items": 2}, "max_batch_items"),
        ({"vad_mode": VadMode.BOTH}, "vad_mode"),
    ],
)
def test_capabilities_reject_invalid_contract(overrides, field):
    with pytest.raises(ValueError, match=field):
        capabilities(**overrides)


def test_registry_identity_drain_switch_and_idempotent_leases():
    async def scenario():
        registry = BackendRegistry()
        await registry.register(capabilities("old"))
        await registry.register(capabilities("new"))
        await registry.mark_ready("old", True)
        first = await registry.acquire(preferred_worker_id="old")
        await registry.mark_ready("new", True)
        await registry.begin_drain("old")
        second = await registry.acquire()
        assert second.worker_id == "new"
        assert (await registry.snapshot())["workers"]["old"]["active_leases"] == 1
        await first.release()
        await first.release()
        await registry.remove("old")
        await second.release()
        return await registry.snapshot()

    snapshot = asyncio.run(scenario())
    assert "old" not in snapshot["workers"]
    assert snapshot["workers"]["new"]["active_leases"] == 0


def test_registry_rejects_identity_change_and_active_removal():
    async def scenario():
        registry = BackendRegistry()
        await registry.register(capabilities())
        with pytest.raises(ValueError, match="immutable model identity"):
            await registry.register(capabilities(model_revision="rev-2"))
        await registry.mark_ready("worker-1", True)
        lease = await registry.acquire()
        with pytest.raises(RuntimeError, match="active leases"):
            await registry.remove("worker-1")
        await lease.release()

    asyncio.run(scenario())


def test_snapshot_excludes_unready_and_draining_workers_from_readiness():
    async def scenario():
        registry = BackendRegistry()
        await registry.register(capabilities())
        initial = await registry.snapshot()
        await registry.mark_ready("worker-1", True)
        ready = await registry.snapshot()
        await registry.begin_drain("worker-1")
        drained = await registry.snapshot()
        return initial, ready, drained

    initial, ready, drained = asyncio.run(scenario())
    assert initial["ready"] is False
    assert ready["ready"] is True
    assert drained["ready"] is False
    assert drained["workers"]["worker-1"]["lifecycle"] == WorkerLifecycle.DRAINING.value
