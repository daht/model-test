# Experimental ASR multi-Pod gateway

This topology runs two independent Qwen3-ASR model processes on one GPU and a
CPU-only gateway in front of them. It is an opt-in capacity experiment. The
default `docker-compose.yml` remains the supported single-owner topology.

The experiment does not change the pinned `qwen-asr==0.0.6`, `vllm==0.14.0`,
or `Qwen/Qwen3-ASR-1.7B` contract. It also does not add stateful batching.

## Behavior

- The gateway probes each backend's `/ready` endpoint.
- A request is admitted only when both configured backends are ready.
- Selection uses the upstream `active_streams` and the gateway's atomic local
  connection counter. Two simultaneous equal-load admissions are reserved on
  different backends.
- An established WebSocket stays on its selected backend until it closes.
- Text frames, binary frames, query parameters, application headers, and the
  upstream terminal close code are proxied without interpreting ASR events.
- `/health` reports gateway liveness. `/ready` reports aggregate readiness and
  per-backend load. The only supported HTTP API forwarding route is
  `GET /v1/transcribe/stream-info`; other `/v1` HTTP requests are rejected. The
  gateway data plane is `/v1/transcribe/stream` WebSocket traffic.
- The gateway never loads a model and has no GPU assignment.

Do not send a client around the gateway directly to a backend. Direct traffic
is visible in upstream load probes, but bypasses the gateway reservation and
stickiness contract.

## GPU sharing prerequisites

Both backends use one Uvicorn worker and one model owner. The Compose file makes
both containers see GPU index `0` and starts each at
`ASR_VLLM_GPU_MEMORY_UTILIZATION=0.35` by default.

This is only a starting value, not proof that two models fit. Before a capacity
run, verify all of the following:

1. Both backends reach `/ready` without CUDA OOM or allocator errors.
2. `nvidia-smi` shows enough steady-state and transient headroom.
3. A simultaneous real streaming warmup succeeds on both backends.
4. Model load plus vLLM cache and peak activations stay below physical VRAM.

Without CUDA MPS, the two CUDA contexts are scheduled by the normal driver and
may mostly time-slice. With MPS, kernels from the two processes may overlap, but
MPS setup and resource limits are host/operator prerequisites. This repository
does not install MPS or configure a Kubernetes device plugin. Run no-MPS and
MPS as separate experiments and retain evidence for each.

On Kubernetes, the standard NVIDIA device plugin normally allocates a whole
GPU to one Pod. Two Pods cannot share it merely by requesting the same GPU.
Configure an approved time-slicing or MPS policy outside this repository first.

## Configure and start

Keep the runtime API credential only in `.env`. Do not put it in Compose,
commands, logs, or test result names.

Render and inspect the topology before starting it:

```bash
cd /opt/model-test
docker compose --env-file .env -f docker-compose.asr-multipod.yml config >/tmp/asr-multipod.rendered.yml
docker compose --env-file .env -f docker-compose.asr-multipod.yml build
```

First perform a one-backend memory feasibility check:

```bash
docker compose --env-file .env -f docker-compose.asr-multipod.yml up -d qwen-asr-backend-1
docker compose --env-file .env -f docker-compose.asr-multipod.yml exec qwen-asr-backend-1 \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/ready').read().decode())"
nvidia-smi
docker compose --env-file .env -f docker-compose.asr-multipod.yml down
```

Then start the full experiment:

```bash
docker compose --env-file .env -f docker-compose.asr-multipod.yml up -d --build
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8002/ready
curl -fsS http://127.0.0.1:8002/v1/transcribe/stream-info
```

The gateway `/ready` response must show `ready_backend_count: 2`. A single
ready backend deliberately returns HTTP 503 so deployment health checks fail
closed.

To rebuild and recreate after a code or environment change:

```bash
docker compose --env-file .env -f docker-compose.asr-multipod.yml build
docker compose --env-file .env -f docker-compose.asr-multipod.yml up -d --force-recreate
curl -fsS http://127.0.0.1:8002/ready
```

## Capacity ladder

Use the same long, continuous speech file and strict real-time client settings
as the single-backend baseline. Supply the credential through the environment:

```bash
read -rsp 'ASR API key: ' API_KEY && echo
export API_KEY
export ASR_AUDIO=/path/to/continuous-speech.wav
```

Run each level at least three times, starting with `12`, `14`, and `16`, then
continue with `20` and `24` only while every previous level passes. This example
keeps evidence outside the repository:

```bash
N=16
RUN_DIR="/tmp/asr-multipod-${N}-$(date +%Y%m%dT%H%M%S)"
mkdir -m 700 "$RUN_DIR"
READY_DIR="$RUN_DIR/ready"
START_GATE="$RUN_DIR/start"
mkdir -m 700 "$READY_DIR"
pids=()
for i in $(seq 1 "$N"); do
  (
    touch "$READY_DIR/$i"
    while [[ ! -e "$START_GATE" ]]; do sleep 0.01; done
    exec python3 scripts/stream_asr_client.py "$ASR_AUDIO" \
      --url ws://127.0.0.1:8002/v1/transcribe/stream \
      --language zh --chunk-ms 200 --realtime --verify-protocol \
      >"$RUN_DIR/client-${i}.log" 2>&1
  ) &
  pids+=("$!")
done
while [[ $(find "$READY_DIR" -type f | wc -l) -lt "$N" ]]; do sleep 0.01; done
touch "$START_GATE"
run_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || run_status=1
done
((run_status == 0))
```

Every client registers in `READY_DIR` before the start file is created, so the
measured run does not depend on how quickly the launch loop creates processes.

Collect backend evidence separately during every run:

```bash
COMPOSE='docker compose --env-file .env -f docker-compose.asr-multipod.yml'
$COMPOSE logs --timestamps --since 1m -f qwen-asr-backend-1 > /tmp/asr-backend-1.log &
$COMPOSE logs --timestamps --since 1m -f qwen-asr-backend-2 > /tmp/asr-backend-2.log &
watch -n 0.5 'curl -fsS http://127.0.0.1:8002/ready; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader'
```

For each backend retain `inference_ms`, `queue_wait_ms`, `lag_debt_seconds`,
`realtime_lag_exceeded`, queue rejections, stream completions, and peak active
streams. Also retain aggregate GPU utilization, VRAM, power, client pass count,
first-partial latency, final latency, and transcript accuracy.

Success is higher strict continuous-speech concurrency without worse latency or
accuracy. GPU utilization reaching 100% by itself is not success. If total
capacity stays near 14 while per-backend inference latency rises, the two
processes are competing for the same A10 compute. If the second backend cannot
become ready, the experiment is blocked by memory feasibility.

## Rollback

Stop only the experimental topology, then restore the unchanged default ASR
service:

```bash
docker compose --env-file .env -f docker-compose.asr-multipod.yml down
docker compose --env-file .env up -d --force-recreate qwen-asr-api
curl -fsS http://127.0.0.1:8002/ready
```

Do not claim a capacity increase or production readiness until the real A10
memory gate, strict protocol ladder, latency/accuracy comparison, and the chosen
no-MPS or MPS configuration have all passed against the same image and commit.
