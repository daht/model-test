# A10 单卡 TTS WebSocket 流式容量测试方案

状态：草案，作为后续实现、测试和结果更新的主文档

创建日期：2026-07-23

最近更新：2026-07-23

## 1. 目标

在同一张 NVIDIA A10 24 GB、同一套服务协议和同一批语料下，评估以下模型的**真实流式音频输出**能力：

- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- `hexgrad/Kokoro-82M` v1.0

最终每个模型必须给出以下结果，不能只给一个含糊的“并发数”：

1. 单请求热态的 TTFA、端到端延迟、流式平滑度和生成速度；
2. 满足既定 SLO 的最大可持续 RPS 和音频 RTFx；
3. 容量点的在途并发 `avg/p95/max`，以及第一个失败负载点；
4. 对应的 GPU/CPU/显存、队列和音频质量证据。

本方案中的 WebSocket 同时指：

- `ws://`：内网或本机明文测试；
- `wss://`：经过 TLS 的生产形态测试。

WS 与 WSS 使用同一应用层协议。容量基线先用 WS 排除 TLS 干扰，最终再用 WSS 做生产链路复测，分别报告结果。

## 2. 本轮范围与公平性边界

### 2.1 主榜：固定音色、流式音频输出

三个模型使用固定、已预热、已缓存的音色条件，采用完全相同的文本集合与到达过程：

| 模型 | 锁定对象 | 本轮音色方式 | 流式准入状态 |
|---|---|---|---|
| Qwen3-TTS | `Qwen3-TTS-12Hz-0.6B-CustomVoice`，记录精确 revision | 官方预置音色 | 需要生产适配器证明能够边生成边返回音频；公开 Python 路径不能直接视为真流式 |
| CosyVoice 3 | `Fun-CosyVoice3-0.5B-2512`，记录模型及 CosyVoice commit | 预注册固定音色 | 官方模型接口具备流式候选能力，但仍须验证 HTTP/WS 层没有缓冲完整音频 |
| Kokoro | `hexgrad/Kokoro-82M` v1.0，记录运行库版本 | 固定 voice | 官方 `KPipeline` 是完整生成一个文本段后再 yield；未实现更细粒度生成前只能标为分段/伪流式 |

只有通过第 5 节“真流式准入”的实现才能进入流式主榜。未通过者保留功能和吞吐结果，但必须明确标为 `non-streaming` 或 `segmented/pseudo-streaming`，不得与真流式 TTFA 横向排名。

### 2.2 独立附表

- 零样本克隆：Qwen 0.6B Base 与 CosyVoice3 使用同一参考音频；Kokoro 记 `N/A`。
- 模型特有能力：VoiceDesign、instruction、cross-lingual 分别测试，不进入共同榜单。
- 后端对比：原生 PyTorch、vLLM、TensorRT、ONNX Runtime 等每种路径单独成行，不混成一个“模型结果”。
- HTTP 完整 WAV：只作为非流式基线，沿用 `scripts/benchmark_tts.py`，不代替本方案的 WS/WSS 测试。

## 3. 被测环境必须锁定

每次测试开始前保存以下信息，任一关键项变化都视为新的测试批次：

- GPU 型号、UUID、驱动版本、功耗上限、时钟策略；
- CPU、内存、操作系统、Docker 和 NVIDIA Container Toolkit 版本；
- 模型仓库、checkpoint、revision/文件哈希；
- 服务 Git commit、容器 image digest；
- Python、PyTorch、CUDA、cuDNN、推理后端及版本；
- dtype、量化方式、FlashAttention/compile/CUDA Graph 等开关；
- worker 数、并发调度方式、队列上限、音色缓存策略；
- 采样参数、输出采样率、声道、样本格式和分块目标；
- 负载机配置、负载机与服务端网络 RTT；
- WS 或 WSS；WSS 还要记录 TLS 终止位置和代理配置。

A10 测试期间必须由当前被测 TTS 服务独占 GPU。ASR、MT 及其他 GPU 进程必须停止，并用进程列表和显存快照留证。

## 4. MiniMax 风格 WebSocket 流式协议

### 4.1 会话顺序

本项目的控制事件和请求结构向 MiniMax WebSocket TTS 靠拢，但不宣称与其 API 完全兼容。保留现有 `/v1/tts/stream` 路径，一个连接在本轮测试中只承载一个 utterance：

```text
client                       server
  |<--- connected_success --------|
  |---- task_start ------------->|
  |<--- task_started -------------|
  |---- task_continue(text) ---->|
  |---- task_finish ------------>|
  |<--- task_continued(audio #0) --|
  |<--- task_continued(audio #1) --|
  |             ...              |
  |<--- task_finished ------------|
  |<--- close code 1000 ----------|
```

鉴权优先使用 WebSocket Upgrade 请求的 `Authorization: Bearer <API_KEY>`；浏览器无法设置该 header 时，使用服务端签发的短时效连接 token，不能把长期 API key 放进 URL。建连成功后服务返回：

```json
{
  "event": "connected_success",
  "session_id": "uuid",
  "trace_id": "uuid",
  "base_resp": {
    "status_code": 0,
    "status_msg": "success"
  }
}
```

客户端用 `task_start` 声明模型、音色和输出格式。字段命名与 MiniMax 对齐；`stream_options.audio_transport` 是本项目扩展：

```json
{
  "event": "task_start",
  "model": "Fun-CosyVoice3-0.5B-2512",
  "voice_setting": {
    "voice_id": "fixed-voice-id",
    "speed": 1.0,
    "vol": 1.0,
    "pitch": 0
  },
  "audio_setting": {
    "sample_rate": 24000,
    "format": "pcm",
    "channel": 1
  },
  "stream_options": {
    "audio_transport": "binary"
  }
}
```

服务接受后返回 `task_started`。客户端随后可发送一个或多个 `task_continue`；本轮公平容量测试只发送一个完整文本事件，增量文本输入另设实验。发送完文本后，以 `task_finish` 明确结束输入：

```json
{"event":"task_continue","text":"待合成文本"}
{"event":"task_finish"}
```

音频输出提供两种传输模式：

- `hex`：MiniMax 兼容形态，`task_continued.data.audio` 为 hex 字符串，便于复用相同事件客户端；
- `binary`：本项目生产和容量测试推荐形态，以 WebSocket binary message 直接传输，避免 hex 导致约 2 倍载荷和额外编解码开销。

`hex` 模式响应示例：

```json
{
  "event": "task_continued",
  "session_id": "uuid",
  "trace_id": "uuid",
  "data": {"audio": "0012ab..."},
  "is_final": false,
  "extra_info": {
    "audio_format": "pcm",
    "audio_sample_rate": 24000,
    "audio_channel": 1,
    "chunk_sequence": 0,
    "sample_offset": 0
  },
  "base_resp": {"status_code": 0, "status_msg": "success"}
}
```

全部音频发送完成后返回 `task_finished`。本项目在 MiniMax 风格字段之外增加压测所需的总样本数、分块数和服务端阶段耗时：

```json
{
  "event": "task_finished",
  "session_id": "uuid",
  "trace_id": "uuid",
  "extra_info": {
    "chunks": 42,
    "total_samples": 80640,
    "audio_sample_rate": 24000,
    "audio_channel": 1,
    "queue_ms": 12.3,
    "synthesis_ms": 2240.1,
    "encode_ms": 3.8
  },
  "base_resp": {"status_code": 0, "status_msg": "success"}
}
```

随后由服务端以 WebSocket close code `1000` 正常关闭连接。协议错误、认证失败、过载、模型错误和超时统一发送 `task_failed`，包含 `session_id`、`trace_id`、稳定的 `base_resp.status_code/status_msg`，再使用对应 close code 关闭。不得把错误音频或空音频当成功。

本项目与 MiniMax 的明确差异只有必要项：模型名称来自本地模型；鉴权沿用本项目安全方案；增加 `binary` 音频传输和观测字段。若未来要求现成 MiniMax SDK 无改动接入，再增加 `/ws/v1/t2a_v2` 兼容路由和严格兼容测试，不提前维护两套路由。

### 4.2 音频帧格式

流式主榜统一使用原始 PCM，避免每块重复 WAV header，也避免容器封装缓存：

- 编码：signed 16-bit little-endian，`pcm_s16le`；
- 声道：mono；
- 采样率：24,000 Hz；
- 每个 WebSocket binary message 对应一个音频 chunk；
- 建议每块承载 40–80 ms 音频，禁止用极小首块人为优化 TTFA。

这里的 24 kHz 是本轮三个开源模型的共同原生输出基线，不代表商业 TTS 行业统一采用 24 kHz。若后续加入腾讯云、火山引擎、MiniMax 等外部服务，必须同时保存两组结果：供应商原生/默认采样率结果，以及解码后统一转换为 24 kHz mono PCM 的链路结果；重采样耗时和质量变化单独统计，不能算作模型生成性能。商业服务采样率调研见 [主流商业 TTS 输出采样率调研](research/commercial-tts-output-sample-rates.md)。

在 `audio_transport=binary` 时，每个 binary message 使用 16 字节固定头，随后紧跟 PCM：

| 偏移 | 长度 | 类型 | 含义 |
|---:|---:|---|---|
| 0 | 4 | ASCII | 固定 magic `TTS1` |
| 4 | 4 | uint32 little-endian | chunk sequence，从 0 连续递增 |
| 8 | 8 | uint64 little-endian | 本块首样本在整段音频中的 sample offset |
| 16 | N | bytes | `pcm_s16le` 音频数据 |

客户端必须检查 magic、sequence 连续性、sample offset 连续性、偶数字节长度和 `task_finished.extra_info.total_samples`。在 `hex` 模式下执行相同检查，sequence 和 offset 取自 `task_continued.extra_info`。WebSocket 自身保证有序传输，但序号和 offset 用于发现服务实现丢块、重复块或错误拼接。

如果未来需要 WAV 流式，必须定义“单个 WAV header + 连续 data + 结束时长度处理”的完整语义，并单独验证播放器兼容性；**一个 chunk 一个完整 WAV 文件**不属于本方案的真流式协议。

## 5. 真流式准入检查

真流式的定义是：模型仍在生成当前 utterance 时，客户端已经收到可播放 PCM；首块到达不能依赖整段音频已经合成完成。

每个“模型 × 后端”必须先通过以下检查，才能执行容量阶梯：

1. 使用至少 30 秒输出的长文本，观察第一块可播放 PCM 在整段完成前到达；
2. 服务端记录首块生成时刻和最终生成完成时刻，满足 `first_chunk_generated < synthesis_finished`；
3. 客户端在收到 `task_finished` 前已连续收到至少两个非空 PCM chunk；
4. chunk 的 sequence、sample offset、总样本数全部一致；
5. 把已接收 PCM 实时写入播放缓冲，能够在连接未结束时开始播放；
6. HTTP/反向代理、ASGI 框架和 WebSocket 层没有把完整音频缓存后一次性发送；
7. 代码路径不得先调用返回完整 WAV/完整 ndarray 的阻塞 `synthesize()`，再人为切块发送。

以下情况均判定为伪流式：

- 合成完整 WAV 后切成多个 WebSocket frame；
- 先完整生成一个长句/文本段，再把该段作为一个 chunk yield；
- 首块只含 header、静音或不足 40 ms 的占位音频；
- 服务只发送一个 binary message；
- 第一块音频与 `task_finished` 几乎同时到达，且服务证据显示合成已结束。

当前仓库已实现 MiniMax 风格事件协议和 `stream_pcm()` 逐块链路；CosyVoice 适配器调用 `inference_zero_shot(..., stream=True)`，通过有界线程队列把 PCM 立即交给 WebSocket，不再聚合完整 WAV。代码级时序测试已证明首块不会等待第二块，但仍须在 A10 真模型部署上保存 `first_chunk_generated < synthesis_finished` 证据后，才能正式标记为通过本准入检查。

## 6. 指标定义

### 6.1 流式客户端指标

- `TTFB`：完整 `task_continue` 发送完成到收到首个服务端音频相关字节，仅作协议诊断。
- `TTFA`：完整 `task_continue` 发送完成到累计收到至少 40 ms 有效 PCM；这是正式首包指标。24 kHz、mono、s16le 下为 960 samples、1,920 bytes PCM。
- `E2E`：完整 `task_continue` 发送完成到收到合法 `task_finished` 且所有音频字节到达。
- `chunk gap`：相邻 PCM frame 在客户端的到达间隔，报告 p50/p95/p99/max。
- `playback underrun`：从 TTFA 后按 24 kHz 消耗虚拟播放缓冲，统计缓冲归零的次数和总时长。
- `single-request RTF`：E2E 墙钟时间除以实际音频时长。
- `RPS`：成功 utterance 数除以测量墙钟时间。
- `audio RTFx`：成功音频总秒数除以测量墙钟时间，越大越好。
- 在途并发：连接已发送 `task_continue` 但尚未收到 `task_finished` 的请求数，报告 avg/p95/max。
- 成功率及各类失败：认证、协议、连接、超时、过载、模型异常、OOM、空音频、损坏、截断分别计数。

延迟报告 p50/p90/p95/p99/max，不能只报平均值。TTFA 只对通过真流式准入的实现有意义。

### 6.2 服务端阶段指标

每个 `request_id` 至少记录：

- 接收 `task_start`、`task_continue`、`task_finish`、进入队列、开始执行、生成首块、生成结束、发送 `task_finished` 的单调时钟；
- queue time、time-to-first-model-chunk、synthesis time、encoding/send time；
- 队列深度、活跃 worker、当前在途数；
- 输入字符/词数、chunk 数、总样本数、实际音频时长；
- 模型、后端、dtype、voice/cache hit。

客户端与服务端机器不共用时钟计算绝对延迟；服务端使用自身单调时钟计算阶段耗时，客户端使用自身单调时钟计算端到端指标。

### 6.3 主机与 GPU 指标

- GPU：显存当前/峰值、SM/engine active、Tensor active、DRAM active、功耗、温度、时钟、throttle、ECC/Xid；
- CPU：总占用、逐核占用、RSS、load、上下文切换；
- 网络：吞吐、重传、连接失败；
- 容器：重启、OOMKilled、退出码；
- 服务：到达率、开始率、完成率、队列深度、活跃请求数。

GPU 监控采样间隔建议 1 秒；容量拐点可增加到 200–500 ms 辅助定位。原始监控数据必须保存，不能只留截图。

## 7. 测试语料

### 7.1 固定分桶

中文和英文分开统计，每种语言至少设置三档；最终边界以实际生成音频时长复核：

| 桶 | 目标音频时长 | 用途 |
|---|---:|---|
| short | 1–3 秒 | 对话首句、短提示，重点观察 TTFA |
| medium | 8–15 秒 | 常规段落，作为主容量负载 |
| long | 30–60 秒 | 验证真流式、长期供给和断流 |

每个桶准备至少 30 条不同文本，保留 `text_id`、语言、字符/词数、预期桶。使用脱敏生产文本时保存抽样规则和随机种子。不得重复单句制造不真实的缓存收益。

主榜采用固定混合比例，初始建议 `short:medium:long = 60%:30%:10%`；上线前必须按真实业务流量确认并更新。单桶结果和混合结果都要保存。

### 7.2 音频正确性门槛

所有容量测试成功样本必须满足：

- PCM 可解码、采样率/声道/位宽正确；
- 非空、非全静音、无 NaN/溢出；
- chunk 连续，无丢块、重复、错序、截断和异常尾音；
- 实际时长处于合理范围；
- 随机抽样试听无明显爆音、断裂、重复或音色跳变；
- 使用固定 ASR 做回译并报告 CER/WER，ASR 版本与参数锁定；
- 支持音色条件的模型报告说话人相似度；加速后端必须与该模型原生基线做质量回归。

质量不合格的响应计为失败，不能进入成功吞吐。

## 8. 负载模型和执行阶梯

### 8.1 冷启动与热态基线

每个模型/后端独立启动，依次执行：

1. 记录模型加载时间和加载峰值显存；
2. 冷态各运行 short/medium/long 1 次，结果单独保存；
3. 至少预热 10 次，直到连续 5 次 TTFA 和 E2E 无明显下降趋势；
4. 热态每桶至少 30 次串行请求，建立单请求基线；
5. 通过真流式准入后才进入并发测试。

冷启动数据不混入热态容量。

### 8.2 闭环并发阶梯

客户端维持固定在途请求数，一个请求完成后立即补发。初始档位：

```text
C = 1, 2, 4, 6, 8, 12, 16, 24, 32
```

每档先预热 2 分钟，再测量 10 分钟。出现拐点后在相邻档位之间补测。闭环用于定位排队和资源拐点，不单独作为最大容量结论。

### 8.3 开环到达率阶梯

使用泊松到达或生产实测到达间隔，不因前一个请求变慢而暂停发新请求：

1. 从单请求可持续速率的约 25% 起步；
2. 每档按 25% 增长，到首次 SLO 失败；
3. 在最后通过点和首次失败点之间按 5%–10% 细分；
4. 每档预热 2 分钟、正式测量 10 分钟；
5. 候选最大容量点运行 30–60 分钟 soak；
6. soak 失败则逐档回退，直到稳定通过。

负载机应与 A10 服务分离，至少使用独立进程；测试期间确认负载机 CPU、网络和事件循环没有饱和。音频边接收边校验并丢弃，抽样才落盘，避免磁盘反压。

## 9. 暂定 SLO 与停止条件

以下是**测试用初始门槛，需产品确认后冻结**，冻结后不得根据测试结果临时放宽：

| 指标 | 暂定门槛 |
|---|---:|
| 成功率 | ≥ 99.9% |
| OOM、容器重启、损坏/空白/截断音频 | 0 |
| short TTFA p95 / p99 | ≤ 500 ms / 800 ms |
| medium/long TTFA p95 / p99 | ≤ 800 ms / 1,200 ms |
| chunk gap p99 / max | ≤ 200 ms / 500 ms |
| playback underrun | 0 |
| 10 分钟档位队列 | 无持续增长趋势 |
| 30–60 分钟 soak | 无显存/RSS 持续增长，无热降频导致的持续性能衰减 |

E2E 与文本/音频长度强相关，正式门槛应按 short/medium/long 分桶确认。在门槛冻结前先完整记录分布，不用一个统一的“P95 ≤ 1 秒”判断所有 TTS 请求。

出现以下任一情况立即停止当前档位并保留现场：

- GPU OOM、Xid、容器退出或服务重启；
- 错误率连续 1 分钟超过 5%；
- 队列超过配置上限或持续增长且无法在测试窗口内回落；
- p99 TTFA 或最大断流超过门槛 2 倍；
- GPU 温度/功耗触发持续 throttle；
- 负载机自身达到 CPU/网络瓶颈；
- 发现系统性空音频、截断、重复或错序。

## 10. 容量判定

最大可持续吞吐是：在冻结的语料分布和 SLO 下，10 分钟开环阶梯通过，并在 30–60 分钟 soak 中再次通过的最高到达率。

对应并发取该容量点的实测在途请求 `avg/p95/max`。可用 Little's Law：

```text
平均在途并发 ≈ 到达率 × 平均端到端时间
```

做一致性检查，但不能替代实测。下列指标都不能单独当作容量：

- 能同时建立多少 WebSocket 连接；
- GPU 利用率达到 100%；
- 显存尚未占满；
- 单请求 RTF 小于 1；
- 固定并发下请求没有立即报错。

## 11. 实施与测试步骤

### 阶段 A：补齐测试前置能力

1. 固化 MiniMax 风格事件协议、hex 兼容模式和 PCM binary 扩展；
2. 把合成器接口改为真正逐块产出 PCM，不再返回完整 WAV 后切块；
3. 分别实现 Qwen3-TTS、CosyVoice3、Kokoro 适配器，并执行真流式准入；
4. 实现 WS/WSS 流式压测客户端，支持闭环和开环两种模式；
5. 实现服务端阶段时间、队列、GPU/主机监控采集；
6. 实现 PCM 连续性、静音、ASR CER/WER 和抽样落盘校验；
7. 为协议解析、断连、超时、服务错误和统计口径添加自动化测试。

当前状态：`app/tts_api.py` 已支持 MiniMax 风格事件、hex 兼容模式及 binary PCM 扩展，`app/tts.py` 已支持 CosyVoice 逐块生成；现有全局锁仍会串行化多个模型推理。`scripts/benchmark_tts.py` 仍只支持 HTTP 闭环整包测试，尚缺 WS/WSS TTFA、chunk gap、开环到达率、服务排队阶段和流式质量工具；Qwen3-TTS 与 Kokoro 也尚未实现真流式适配器。

### 阶段 B：服务器准备

1. 停止 ASR、MT 和其他 GPU 容器，确认 A10 上只剩当前 TTS 进程；
2. 启动单一被测模型，等待 health ready，而不只是进程存活；
3. 保存环境清单、容器 inspect、镜像 digest、Git 状态和 GPU 空闲快照；
4. 从负载机分别验证 WS 和 WSS 握手、认证、协议顺序与正常关闭；
5. 使用 long 文本执行真流式准入并保存逐帧时间线；
6. 预热并验证服务端阶段指标与 GPU 监控时间范围对齐。

### 阶段 C：正式负载

1. 单请求 short/medium/long 基线；
2. 闭环并发阶梯定位拐点；
3. 开环到达率阶梯定位最高通过点与首次失败点；
4. 最高通过点做 30–60 分钟 soak；
5. 重启服务，至少重复一次容量点，检查可复现性；
6. 对音频做自动校验和固定比例人工抽检；
7. 切换到下一个模型，恢复完全相同的语料与负载流程。

### 阶段 D：WSS 生产链路复测

在 WS 基线完成后，使用实际反向代理、TLS 和公网/生产等价网络，复测：

- 单请求基线；
- WS 容量点的 80%、100%、110%；
- 候选容量点 30 分钟稳定性；
- 代理超时、最大连接数、idle timeout、buffering 和断连传播。

WS 与 WSS 的容量、TTFA 和断流结果分表，不能覆盖。

## 12. 证据目录和产物

每次测试使用不可覆盖的批次目录：

```text
log/tts-stream/<UTC批次时间>/
├── manifest.json                 # 环境、模型、代码、配置、语料哈希
├── protocol-preflight.json       # 真流式准入及逐帧证据
├── client-observations.jsonl     # 每请求/每 chunk 客户端记录
├── server-events.jsonl           # 服务阶段事件
├── gpu.csv                       # GPU 原始监控
├── host.csv                      # CPU/内存/网络监控
├── levels.json                   # 各负载档聚合结果
├── report.md                     # 人读报告
├── audio-samples/                # 固定抽样音频
└── quality.json                  # PCM、CER/WER、相似度与抽检结果
```

API key、参考音频中的敏感信息和生产原文不得写入证据；`manifest.json` 只保存脱敏 ID 或哈希。

## 13. 报告模板

| 模型/checkpoint | 后端/dtype | WS/WSS | 真流式通过 | 语料桶/语言 | 到达率 RPS | 在途 avg/p95/max | 成功率 | TTFA p50/p95/p99 | gap p95/p99/max | E2E p50/p95/p99 | 音频 RTFx | 峰值显存 | SLO |
|---|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---|
| 待实测 | 待实测 | WS | 否/是 | 待实测 | — | — | — | — | — | — | — | — | — |

每个模型的结论页还必须包含：

- 单请求冷态与热态结果；
- 最大可持续容量点和第一个失败点；
- 主要瓶颈证据，而不是推测；
- 音频质量结果；
- 相对 HTTP 整包基线和 WS/WSS 的差异；
- 已知限制和未覆盖场景；
- 原始证据路径与复现所需版本信息。

## 14. 当前结论与下一步

目前还不能给出 A10 上三种模型的可靠并发数字。现有 CosyVoice 单请求结果只证明完整 WAV 推理可用，不证明真流式或并发容量；现有全局锁还可能把客户端并发退化为串行排队。

下一步按顺序完成：

1. 在 A10 真模型部署上验证 CosyVoice3 首块时序、连续性和长文本播放；
2. 实现 WS/WSS 流式负载与监控工具；
3. 对 Qwen 和 Kokoro 做流式可行性验证，通过后再进入共同榜；
4. 冻结业务 SLO 与真实语料分布；
5. 按本方案执行三模型 A10 独占测试并持续更新本文档及结果报告。

## 15. 参考资料

模型能力、官方接口限制、A10 规格和完整研究依据见：

- [A10 单卡 TTS 并发吞吐评估方法](research/tts-a10-concurrency-benchmark-methodology.md)
- [现有 HTTP TTS 容量与成本压测](tts-capacity-cost-benchmark.md)

## 16. 更新记录

| 日期 | 变更 | 作者 |
|---|---|---|
| 2026-07-23 | 实现 MiniMax 风格事件协议、hex/binary PCM 输出和 CosyVoice3 逐块生成；更新剩余缺口 | Codex |
| 2026-07-23 | 初版：确定 WS/WSS 真流式协议、准入门槛、测试阶梯、指标和产物结构 | Codex |
