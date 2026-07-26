# WebSocket 流式 TTS 压测执行说明

本轮先评估 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 在 A10 单卡上的真实流式容量。测试分为客户端负载和 A10 服务端监控，两端时间范围必须重叠。

2026-07-24 首轮闭环/开环扫描与服务端瓶颈分析已完成，结果见 [CosyVoice 3 在 NVIDIA A10 上的 WebSocket 流式容量与瓶颈评估](tts-cosyvoice3-a10-capacity-evaluation-2026-07-24.md)。当前实现因全局推理锁和约 4 秒模型输出块，尚未达到临时交互 SLO；本文保留为优化后复测的执行入口。

## 工具

- `scripts/benchmark_tts_stream.py`：从独立客户端发送 MiniMax 风格 WebSocket 请求，支持闭环并发和开环泊松到达率。
- `scripts/monitor_tts_benchmark.sh`：在 A10 服务器采集 GPU、GPU 进程、容器资源、服务日志、版本和容器状态。
- `docs/tts-stream-benchmark.jsonl`：首批中文短、中、长诊断语料，每档 4 条。该集合用于寻找初步拐点；正式报告前按测试方案扩充至每档至少 30 条。

客户端逐请求记录以下指标：

- TTFA、E2E、请求成功率；
- chunk 数量、最大 chunk gap；
- sequence 和 sample offset 连续性；
- 实际音频秒数、音频 RTFx；
- 模拟播放缓冲断流时长；
- 实测平均在途请求数。

## 1. A10 服务端启动监控

服务端拉取包含监控脚本的提交后执行：

```bash
cd /opt/model-test
scripts/monitor_tts_benchmark.sh
```

看到以下提示后保持终端运行：

```text
TTS benchmark monitor started.
Evidence directory: /tmp/tts-monitor/runs/<run-id>
Run the benchmark, then press Ctrl+C.
```

监控脚本不会发送 TTS 请求，也不读取 `API_KEY`。

## 2. 客户端预检

使用独立负载机执行短时间预检：

```bash
export API_KEY='<部署密钥>'
export TTS_STREAM_BENCHMARK_URL='ws://118.195.185.141:8003/v1/tts/stream'

.venv/bin/python scripts/benchmark_tts_stream.py \
  --corpus docs/tts-stream-benchmark.jsonl \
  --output-dir log/tts-stream-preflight \
  --concurrency 1 \
  --duration-seconds 30 \
  --warmup-requests 1
```

预检必须满足：无协议错误、无损坏音频、sequence/offset 连续，并且每个请求至少收到一个音频块。预检结果不能作为容量结论。

## 3. 闭环并发阶梯

初步阶梯：

```bash
.venv/bin/python scripts/benchmark_tts_stream.py \
  --corpus docs/tts-stream-benchmark.jsonl \
  --output-dir log/tts-stream-closed \
  --concurrency 1,2,4,6,8 \
  --duration-seconds 120 \
  --warmup-requests 2 \
  --request-timeout-seconds 180
```

当前 CosyVoice 适配器存在全局推理锁，预计并发增加后主要表现为排队和 TTFA 上升；闭环测试用于确认该拐点，不能单独作为最大到达率结论。

## 4. 开环到达率阶梯

根据闭环并发 1 的实测 RPS 选择初始到达率。第一轮建议从较低值开始：

```bash
.venv/bin/python scripts/benchmark_tts_stream.py \
  --corpus docs/tts-stream-benchmark.jsonl \
  --output-dir log/tts-stream-open \
  --concurrency '' \
  --arrival-rates 0.03,0.05,0.08,0.10 \
  --duration-seconds 300 \
  --warmup-requests 2 \
  --request-timeout-seconds 300
```

达到首次失败点后，在最后通过点与首次失败点之间补 5%–10% 档位。正式容量候选点再运行 30–60 分钟 soak。

## 5. 结束监控并提供证据

所有客户端档位结束后，在服务端监控终端按 `Ctrl+C`。脚本会输出归档路径：

```text
TTS monitor archive: /tmp/tts-monitor/runs/<run-id>.tar.gz
```

需要提供的唯一服务端文件就是该 `tar.gz`。它包含：

```text
metadata.json
gpu.csv
gpu-processes.csv
container.csv
service.log
collector-errors.log
report.json
report.md
manifest.sha256
```

客户端保留三个输出文件：

```text
tts-stream-benchmark.json
tts-stream-observations.jsonl
tts-stream-benchmark.md
```

API key 不会写入上述报告。服务日志中的 IPv4 地址会在归档前脱敏。

## 6. 初始判定门槛

脚本当前默认使用测试方案中的初始门槛：

- TTFA p95 ≤ 0.8 秒；
- chunk gap p99 ≤ 0.5 秒；
- 错误率 ≤ 0.1%；
- 播放断流总时长为 0。

这些门槛尚待产品确认。当前已观察到 CosyVoice 单请求 TTFA 约 3–4 秒、chunk gap 约 2.4–3.1 秒，因此预计功能正确但会暂时判定 SLO 失败；这正是本轮瓶颈定位需要量化的问题。

## 7. 方案 B：官方 Triton + TensorRT-LLM 引擎部署与复测路径

为了打破 Python 进程内全局锁造成的串行推理瓶颈，服务后端确定采用 **NVIDIA 官方 CosyVoice3 Triton + TensorRT-LLM 部署方案**（官方来源：[CosyVoice runtime/triton_trtllm](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)）。

### 7.1 服务端部署命令

1. **拉取官方预建 Triton+TRT-LLM 镜像**：
   ```bash
   docker pull soar97/triton-cosyvoice:25.06
   ```

2. **启动容器并自动完成阶段 0~3 (权重转换、TRT 引擎编译与 Triton 服务拉起)**。
   依赖通过阿里云 PyPI 镜像安装，模型下载通过 Hugging Face 镜像；生成的权重、引擎和 model repository 会保存在宿主机挂载的 CosyVoice 目录：
   ```bash
   docker run -d --name "cosyvoice-triton-server" \
     --gpus all \
     --net host \
     -e HF_ENDPOINT="https://hf-mirror.com" \
     -e PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple" \
     -e PIP_TRUSTED_HOST="mirrors.aliyun.com" \
     -v /opt/model-test:/workspace/model-test \
     --shm-size=4g \
     soar97/triton-cosyvoice:25.06 \
     bash -lc "set -e; pip3 install --no-cache-dir x_transformers s3tokenizer; cd /workspace/CosyVoice/runtime/triton_trtllm; bash run_cosyvoice3.sh 0 3"
   ```

3. **验证 gRPC 服务就绪**：
   当前官方脚本使用 HTTP `18000`、gRPC `18001`、metrics `18002`。观察 `docker logs -f cosyvoice-triton-server`，确认 Triton 已启动后，再让 FastAPI/WS gRPC adapter 连接 `127.0.0.1:18001`。不要把 Triton gRPC 端口误当成外部 WebSocket 压测地址；压测仍应访问 adapter 的 `/v1/tts/stream`。
