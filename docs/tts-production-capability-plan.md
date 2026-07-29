# TTS 生产能力分层与高吞吐路线

本计划把 TTS 拆成两条生产线：

1. `interactive_stream`：面向在线交互，要求首包和分块连续性。
2. `bulk_synthesis`：面向高吞吐批量生成，优先吞吐和队列稳定性，不把 chunk gap 当主指标。

这两条线不能共用同一组 SLO。把流式 SLO 强行套到批量吞吐，会把系统尾延迟放大成错误的容量上限。

## 1. 现状判断

仓库里已经有三种可用后端：

- `cosyvoice`：当前实现有全局锁，适合作为兼容/回滚路径，不适合作为高吞吐主线。
- `triton`：官方 Triton + TensorRT-LLM 路线，适合真实流式和更好的服务端并行。
- `qwen` / `vllm_omni`：已经有显式微批或连续 batching 的入口，适合把吞吐从“单请求串行”提升到“队列 + 批处理”。

现有压测结果说明的是：

- `vllm_omni` 流式路径在当前 SLO 下只能作为实时交互线继续调优。
- 这不等于 TTS 只能做到这么低的吞吐。
- 真正的高吞吐应走批量合成或服务端 batching，而不是继续把实时流式指标硬抬高。

## 2. 生产能力分层

### 2.1 交互流式线

用途：

- 聊天回复
- 语音助手
- 需要首包快、播放不断的场景

推荐接口：

- `POST /v1/tts/stream`

推荐后端：

- `vllm_omni`
- `triton`

主指标：

- `TTFA p95`
- `chunk gap p99`
- 播放断流

这条线的目标是“稳定实时”，不是“极限吞吐”。

### 2.2 高吞吐批量线

用途：

- 长文本批量生成
- 离线预生成
- 合成缓存
- 业务侧可排队的请求

推荐接口：

- `POST /v1/tts`

推荐后端：

- `qwen`
- `triton` 的批量/非流式部署形态

主指标：

- requests/sec
- audio-seconds/sec
- queue wait p95
- completion latency p95/p99
- `/v1/tts/capacity` 中的 queue depth、batch size、last batch size

这条线不要用 `chunk gap` 定容量。它应该以“每张 A10 每秒能稳定产出多少音频秒”为主。

## 3. 推荐部署拓扑

单实例只加载一个模型，因此生产上建议至少拆成两个实例：

1. `tts-stream` 实例：只服务交互流式。
2. `tts-batch` 实例：只服务高吞吐批量。

前置路由可以按以下规则做：

- `/v1/tts/stream` 始终进流式实例。
- `/v1/tts` 默认进批量实例。
- 长文本或可异步的任务一律不进流式实例。

如果业务只有一台 A10，先保住批量实例，再把流式实例的并发压低，避免互相抢 GPU。

## 4. 优化顺序

### 第一步：切分产品目标

把“实时播报”与“高吞吐生成”拆成两个产品目标，不再共用一个 SLO。

### 第二步：把高吞吐主线迁到批量后端

优先使用 `qwen` 或 `triton` 的批量路径，把请求聚合窗口、队列和超时放到后端前面。
上线前通过 `GET /v1/tts/capacity` 确认当前实例是否支持 micro-batch、当前 batch
参数和队列深度。

### 第三步：单独调批量参数

批量线的初始调参只改这几个：

- `tts_qwen_batch_size`
- `tts_qwen_batch_wait_ms`
- `tts_qwen_queue_size`

不要先放宽流式 SLO。

### 第四步：单独调流式参数

流式线只改：

- `codec_chunk_frames`
- `codec_chunk_ramp`
- vLLM-Omni 或 Triton 的 backend 侧并发/instance 配置

### 第五步：再做横向扩容

单卡稳定后，再复制实例，不要先把单实例 queue 拉大到失控。

## 5. 现阶段结论

- 现在的 `0.30 RPS` 结果只说明：当前流式实现下，单张 A10 的实时播放上限大约在这个量级。
- 它不代表 TTS 这个问题在 A10 上就只能到这里。
- 生产级高吞吐的正确做法，是把实时和批量分层，然后让批量线吃 batching，流式线保实时。
