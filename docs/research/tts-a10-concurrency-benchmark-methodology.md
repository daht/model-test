# A10 单卡 TTS 并发吞吐评估方法

更新日期：2026-07-23

## 结论先行

目前没有任何第一方资料给出 **NVIDIA A10 单卡**运行 Qwen3-TTS、Fun-CosyVoice3-0.5B-2512 或 Kokoro-82M 的可持续并发数。因此不能从参数量、单请求 RTF 或其他显卡结果换算出 A10 的可靠数字；三者的容量都必须在目标 A10、目标服务代码、目标推理后端和真实请求分布上实测。

“并发能力”也不是脱离业务 SLO 的一个固定值。本项目应报告：在指定请求组合下，同时满足错误率、首包延迟、完成延迟、稳定性和音频正确性门槛的 **最大可持续到达率**及其对应的 **在途并发数**。请求/秒只适合相同文本长度的测试；跨长度、跨模型的主吞吐指标应使用“每秒生成的音频秒数”。

A10 的官方规格是 24 GB GDDR6、600 GB/s 显存带宽、150 W TDP，并支持 FP16/BF16 Tensor Core；这些只确定测试边界，不能推出模型容量。[NVIDIA A10 官方规格](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)

## 被测对象必须先锁定

名称相近的 checkpoint、功能和后端成本不同。每份结果都必须记录模型仓库、精确 revision/文件哈希、运行时 Git commit、容器镜像 digest、PyTorch/CUDA/cuDNN、推理后端及版本、dtype、编译选项、采样参数、输出采样率和服务 worker 数。

截至本次审计，Qwen 官方 12Hz 系列公开了 5 个 checkpoint：`Qwen3-TTS-12Hz-1.7B-VoiceDesign`、`Qwen3-TTS-12Hz-1.7B-CustomVoice`、`Qwen3-TTS-12Hz-1.7B-Base`、`Qwen3-TTS-12Hz-0.6B-CustomVoice`、`Qwen3-TTS-12Hz-0.6B-Base`；官方 Python 包源码版本为 0.1.1。[Qwen 官方 collection](https://huggingface.co/collections/Qwen/qwen3-tts) [qwen-tts 0.1.1 pyproject](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/pyproject.toml) Kokoro 应锁定 v1.0（2025-01-27）及官方运行库版本 0.9.4。[Kokoro releases](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/README.md#releases) [kokoro 0.9.4 pyproject](https://github.com/hexgrad/kokoro/blob/dfb907a02bba8152ca444717ca5d78747ccb4bec/pyproject.toml) CosyVoice 模型锁定为 2025-12 发布的 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，运行时仍须记录实际部署的 CosyVoice Git commit。[CosyVoice roadmap](https://github.com/FunAudioLLM/CosyVoice#roadmap)

| 对象 | 本次应锁定的版本与能力 | 官方推理/流式接口 | 官方支持的加速路径 | 原生输出 | 官方披露的性能 |
|---|---|---|---|---|---|
| Qwen3-TTS | 必须写明具体 checkpoint；官方发布同时包含 0.6B/1.7B、12Hz 的 Base、CustomVoice、VoiceDesign，不能笼统写“Qwen3-TTS” | Python 包有 `generate_custom_voice`、`generate_voice_design`、`generate_voice_clone` 和 list batch 输入。重要限制：当前公开 API 的 `non_streaming_mode=False` 只模拟流式文本输入，源码明确不启用真实 streaming input/generation；不能据此测本地真流式 TTFA | 官方建议 FlashAttention 2 以降低显存；vLLM-Omni 已有离线单样本和 batch 推理，但官方 README 明确当前尚无 online serving，故不能把离线 batch 当在线并发服务 | Tokenizer 官方配置为输入/输出 24,000 Hz；仍应从 API 返回的 `(wav, sample_rate)` 记录实际值 | 技术报告披露并发 1/3/6 数据，但硬件只写“single typical computational resource”且使用内部 vLLM V0 与 tokenizer compile/CUDA Graph；不能作为 A10 容量值 |
| Fun-CosyVoice3-0.5B-2512 | 明确使用 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，不要混用 1.5B、RL checkpoint 或 CosyVoice2 | `AutoModel`/`CosyVoice3` 提供 zero-shot、cross-lingual、instruct；接口有 `stream` 参数，官方描述同时支持文本输入与音频输出双流 | `CosyVoice3` 构造器支持 `fp16`、`load_vllm`、`load_trt`；官方仓库支持 vLLM 0.11.x+ V1 或 0.9.0，其他版本未测试；TensorRT engine 有并发参数，FP16 DiT engine 官方代码提示存在性能问题，必须单独验质 | 模型 `cosyvoice3.yaml` 明确 `sample_rate: 24000` | 官方宣称双流延迟最低 150 ms，没有 A10 或并发测试条件；官方公开的 CER/WER/说话人相似度是质量指标，不是吞吐 |
| Kokoro-82M | 锁定 `hexgrad/Kokoro-82M` v1.0（82M）；不要混入旧版 kLegacy 或非官方量化移植 | 官方 `KPipeline` 返回 generator，但源码是先切句/切段、完整生成一段后再 yield；不是模型原生因果音频流，也没有现成多请求 batch serving API | 主路径为 PyTorch，`device=None` 自动优先 CUDA；官方仓库另有 ONNX export/ONNX Runtime 示例及 Triton-compatible 转换脚本，但没有第一方吞吐基准。PyTorch 与 ONNX 结果必须分表 | 官方示例以 24,000 Hz 保存音频 | 官方模型卡只做“轻量、较快”等定性陈述，没有 A10 并发、TTFA 或吞吐数字；全部容量指标必须实测 |

来源：

- Qwen3-TTS：[官方模型/功能表、Python 用法与 FlashAttention 2 建议](https://github.com/QwenLM/Qwen3-TTS#quickstart)、[公开 Python API 的流式限制源码](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py#L470-L515)、[Tokenizer 24 kHz 配置](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/blob/main/config.json)、[官方 vLLM-Omni 说明](https://github.com/QwenLM/Qwen3-TTS#vllm-usage)、[技术报告](https://arxiv.org/pdf/2601.15621)。
- Fun-CosyVoice3：[官方模型卡](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)、[官方示例](https://github.com/FunAudioLLM/CosyVoice/blob/main/example.py)、[`CosyVoice3` 后端开关源码](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py)、[24 kHz 配置](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/blob/main/cosyvoice3.yaml)、[官方 vLLM 版本说明](https://github.com/FunAudioLLM/CosyVoice#vllm-usage)、[双流最低 150 ms 声明](https://github.com/FunAudioLLM/CosyVoice#key-features)。
- Kokoro：[官方 v1.0 模型卡与 82M 说明](https://huggingface.co/hexgrad/Kokoro-82M)、[官方 KPipeline 用法](https://github.com/hexgrad/kokoro#usage)、[CUDA 设备选择与分段 yield 源码](https://github.com/hexgrad/kokoro/blob/dfb907a02bba8152ca444717ca5d78747ccb4bec/kokoro/pipeline.py#L358-L405)、[官方 ONNX export 示例](https://github.com/hexgrad/kokoro/blob/dfb907a02bba8152ca444717ca5d78747ccb4bec/examples/export.py)。

Qwen 技术报告的流式效率表可作为趋势核验，不可当作 A10 预测：12Hz-0.6B 在并发 1/3/6 时报告首包 97/179/299 ms、RTF 0.288/0.338/0.434；12Hz-1.7B 为 101/195/333 ms、RTF 0.313/0.363/0.463，每包 4 token（320 ms 音频）。该实验没有披露 GPU 型号，且使用内部 vLLM V0、tokenizer `torch.compile` 与 CUDA Graph；公开 `qwen-tts` Python 路径不能等价复现。[Qwen3-TTS 技术报告 Efficiency/Table 1](https://arxiv.org/pdf/2601.15621)

## 公平比较的场景矩阵

不要用三个模型各自“最方便的 demo”横比。至少拆成以下独立榜单，每一格使用完全相同的文本集合和负载过程：

1. **标准音色、非流式**：Qwen CustomVoice、CosyVoice 固定且预注册的说话人条件、Kokoro 固定 voice。模型初始化和可复用音色预处理不计入请求延迟。
2. **标准音色、流式音频输出**：只测试运行代码真实支持的流式路径；第一块可播放 PCM 到达客户端才算 TTFA。当前公开 Qwen Python API 不提供真流式生成，Kokoro 是整段生成后 yield；两者若由服务自行切段，只能标“分段/伪流式”，不可与 CosyVoice 双流 TTFA 放在同一榜单。若函数内部 generator 存在、HTTP 层仍缓冲完整 WAV，也必须标成非流式。
3. **零样本克隆、非流式**：Qwen Base 与 CosyVoice3 使用同一条符合各自要求的参考音频和参考文本；分别报告“每请求重算参考特征”和“按 speaker-id 缓存参考特征”。Kokoro v1.0 官方接口不是同等零样本克隆能力，应记 N/A，不能拿预置 voice 冒充。
4. **模型特有能力**：VoiceDesign、instruction、cross-lingual 只能各自另表，不能进入共同吞吐排名。

每个榜单再分别测“官方原生基线”和“该模型官方支持的最佳后端”。不同后端的数字不能混成一个模型数字。例如 Qwen vLLM-Omni 当前是离线推理，不应以其 batch throughput 宣称在线 HTTP 并发；CosyVoice 的 vLLM/TensorRT 也应与原生 PyTorch 分表。

## 测试集与负载

### 固定语料

- 中文与英文分开报告；如生产流量有日语等，再按真实权重加入。不要把不同语言混成无法解释的均值。
- 建立短、中、长三组固定文本，并保存文本 ID、字符/词数。建议从生产脱敏样本按最终音频时长分桶，而不是仅按字符数分桶，因为语速、标点和语言都会改变计算量。
- 每组至少准备足够多的不同文本，避免重复同一句触发不真实的缓存收益。随机种子和生成参数固定，但不能因固定 seed 而复用生成结果。
- 记录每个成功响应的实际音频时长；吞吐用实际音频时长计算。编码/重采样如属于生产响应就保留在端到端测试内，但另做一个原始 PCM 推理层测试以定位瓶颈。
- 先离线校验所有文本能正常读出，无空音频、截断、异常时长或明显不可懂内容；容量测试不能把错误音频计作成功吞吐。

### 两种负载模型

1. **闭环并发阶梯**用于观察并发 C：客户端每完成一个请求立即发下一个。建议从 `C=1` 开始，逐级加到出现 OOM、错误或 SLO 失败；接近拐点时用更细粒度补点。它回答“固定在途并发下表现如何”，但会受延迟反馈限制，不能单独代表最大到达率。
2. **开环到达率阶梯**用于测容量：按泊松或生产实测到达间隔发送，逐级提高请求/秒，不因上一个请求变慢而停止发新请求。它能暴露排队失稳。每一级先预热，再测至少 10 分钟；候选容量点做至少 30–60 分钟稳态/浸泡测试。

负载发生器必须在另一进程，最好在另一台机器；保持连接复用。客户端要有足够大的连接池和自身 CPU 余量。生成音频应边计长度/哈希边丢弃或写到独立高速盘，避免压测机 I/O 反压服务。

## 必须采集的指标

### 客户端端到端指标

- 成功请求率及超时、HTTP/WS 错误、OOM、取消、空音频、截断分别计数。
- `TTFA`：请求发送完成到第一个可播放音频字节到达；仅对真正流式响应有意义。
- `E2E latency`：请求开始到最后一个音频字节到达，报告 p50/p90/p95/p99/max，不能只报均值。
- 排队时间、模型执行时间、编码/传输时间：由服务端时间戳拆分；否则无法判断并发失败在队列还是 GPU。
- 请求吞吐 `RPS = 成功请求数 / 测量墙钟时间`。
- 音频吞吐 `RTFx = 成功音频总秒数 / 测量墙钟时间`，越大越好；等价的聚合实时因子 `RTF = 测量墙钟时间 / 成功音频总秒数`，越小越好。单请求 RTF 的百分位也应报告，但不能对各请求 RTF 做简单平均代替聚合吞吐。
- 流式稳定性：块间间隔 p50/p95/p99、最大断流间隙、首包大小和后续块大小。只看 TTFA 会掩盖首包之后卡顿。

### 服务与主机指标

- GPU：显存当前值/峰值、GPU engine/SM active、Tensor Core active、DRAM active、功耗、温度、时钟、throttle reason、ECC/Xid。
- CPU：总占用及逐核占用、RSS、load、上下文切换；文本前端、音频编码和参考音频处理可能先成为瓶颈。
- 队列：到达率、开始服务率、完成率、队列深度、在途数、各 worker 活跃数。

NVIDIA 官方建议用 DCGM/DCGM Exporter 采集 GPU 指标；其 profiling 指标包括 engine activity、Tensor active、DRAM active，默认 1 Hz 且可配置到更高频率。[DCGM profiling 指标定义](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics) [DCGM Exporter](https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html) 简单单机复核也可使用官方 `nvidia-smi dmon` 的功耗、利用率、时钟和显存组。[nvidia-smi dmon 官方文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html#device-monitoring)

## 容量判定

测试前先由业务给出 SLO，示例阈值不能事后按结果修改。建议判定框架如下（具体数字由业务确认）：

- 成功率不低于目标，且 OOM、进程重启、空音频、截断为 0；
- 流式场景 TTFA p95/p99 与最大块间断流满足交互要求；
- 非流式 E2E p95/p99 满足对应短/中/长文本要求；
- 开环测试中队列深度和延迟不随时间持续增长；
- 音频 RTFx 在测量窗口稳定，无持续降频或显存增长；
- 抽样音频通过时长、可解码、非静音和 ASR 可懂度检查；加速后端还需与原生基线做质量回归。

**最大可持续吞吐**是通过上述门槛的最高开环到达率；**可承载并发**是该点实测的在途请求分布（至少报告平均、p95、最大），也可用 Little's Law `平均并发 ≈ 到达率 × 平均响应时间` 做一致性校验，但不能用它替代实测。第一个失败点也要保留，以展示容量边界。

不要把“显存还没满”“GPU 利用率 100%”“单请求 RTF < 1”或“能接受 C 个连接”单独当作并发容量结论。服务可能串行排队，模型不同阶段也可能分别受 CPU、显存带宽或 GPU 计算限制。

## 推荐执行顺序

1. 清空 A10 上其他 GPU 工作负载；记录 GPU UUID、驱动、功耗上限、时钟策略、主机 CPU/内存。三个模型使用同一台机器并依次独占 GPU。
2. 每个“模型 × checkpoint × 后端 × dtype × 场景”启动独立进程；加载后完成固定次数预热，冷启动时间和加载峰值显存另表，不混入热态容量。
3. 做单请求短/中/长基线，验证输出、TTFA/E2E、音频时长、峰值显存；重复至少 30 次以观察生成随机性。
4. 做闭环并发阶梯找到拐点；每级使用相同语料分布和时长，监控服务队列与 GPU/CPU。
5. 围绕拐点做开环到达率阶梯；失败点回退一档做 30–60 分钟 soak，再向上细分确认边界。
6. 在最佳候选后端重复同样流程，并做音频质量回归。任何调参变化都生成新结果行，不能覆盖原始结果。
7. 最终输出共同场景榜单和模型特有场景附表；每项给出均值与尾部、95% bootstrap 置信区间、原始 JSON/CSV 路径以及失败原因。

## 结果表模板

| 模型/checkpoint | 后端/dtype | 场景 | 文本桶/语言 | 到达率 RPS | 在途 avg/p95/max | 成功率 | TTFA p50/p95/p99 | E2E p50/p95/p99 | 音频 RTFx | 峰值显存 | GPU/CPU 瓶颈证据 | SLO 通过 |
|---|---|---|---|---:|---|---:|---|---|---:|---:|---|---|
| 待实测 | 待实测 | 待实测 | 待实测 | — | — | — | — | — | — | — | — | — |

每个模型最终应明确给出三个数字，而不是一个含糊的“并发数”：

1. 单请求热态性能（短/中/长的 TTFA、E2E、RTF）；
2. 满足 SLO 的最大可持续 RPS 与音频 RTFx；
3. 该容量点对应的在途并发 avg/p95/max，以及第一个失败负载点。

## 哪些内容必须实测

- A10 上每个 checkpoint/后端/dtype 能否装入 24 GB，以及静态/峰值显存；
- 单请求和并发下的 TTFA、E2E、块间间隔、请求 RPS、音频 RTFx；
- 最大可持续到达率、在途并发、排队拐点和 30–60 分钟稳定性；
- 现有 API 是否真正并行调用 GPU，是否被全局锁、单 worker 或完整 WAV 缓冲串行化；
- prompt/音色特征缓存的收益与内存代价；
- FlashAttention 2、vLLM-Omni、CosyVoice vLLM/TensorRT、Kokoro ONNX Runtime 等每条候选后端在 A10 上的兼容性、性能和质量；
- 输出采样率、音频有效性、截断/静音、加速前后可懂度与音色质量；
- CPU 文本前端、音频编码、网络和负载发生器是否成为瓶颈。

官方的 97 ms（Qwen3-TTS）和 150 ms（CosyVoice）没有本项目 A10 并发测试条件，Kokoro 也没有第一方 A10 容量结果。三者最终“是多少”只能由以上实测表回答，不能合理外推。
