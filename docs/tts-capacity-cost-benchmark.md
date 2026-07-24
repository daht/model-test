# TTS 容量与成本压测

> 本文只覆盖 HTTP 完整 WAV 基线。WS/WSS 真流式协议、TTFA、分块平滑度和 A10 三模型容量评估见 [A10 单卡 TTS WebSocket 流式容量测试方案](tts-a10-websocket-streaming-test-plan.md)。

脚本 `scripts/benchmark_tts.py` 对已部署的 TTS HTTP 接口
`POST /v1/tts` 做真实并发压测。每条语料是一行 JSON：

```json
{"text":"你好，欢迎使用我们的产品。","voice":"default"}
```

`voice` 可省略。脚本会在每个并发档持续发送请求，记录请求延迟、成功率、文本字符吞吐，并从返回 WAV 的 header 计算实际生成音频秒数。成本按音频秒计算，避免不同文本长度导致的请求数偏差。

```bash
export API_KEY='部署时的密钥'
export TTS_BENCHMARK_URL='http://127.0.0.1:8003/v1/tts'
python scripts/benchmark_tts.py \
  --corpus docs/tts-benchmark.jsonl \
  --output-dir log/tts-capacity \
  --concurrency 1,2,4,8,16 \
  --duration-seconds 30 \
  --a10-monthly-cost-cny 2132.72
```

输出 `tts-benchmark.json`（机器可读）和 `tts-benchmark.md`（报告）。默认 SLO 是 P95 ≤ 1 秒、错误率 ≤ 0.1%；可用命令行参数覆盖。`selected_sustainable` 是满足 SLO 的最高并发档。

核心公式：

```text
月音频秒 = 音频秒/秒 × 2,592,000
每百万音频秒 GPU 成本 = A10 月成本 × 1,000,000 ÷ 月音频秒
```

Mock 后端只能验证压测链路，不能作为商业容量证据。正式结论应在独占 GPU、固定模型/采样率/音色和已预热服务上运行，并结合 GPU 监控记录。
