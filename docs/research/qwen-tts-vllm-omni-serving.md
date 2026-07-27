# Qwen3-TTS 与 vLLM-Omni 服务方式核查

核查时间：2026-07-27  
上游版本：

- Qwen3-TTS：`022e286b98fbec7e1e916cb940cdf532cd9f488e`（`QwenLM/Qwen3-TTS`，2026-03-17）
- vLLM-Omni：`8001bb155dae5798a1ae891ae2529a314c6ee99a`（`vllm-project/vllm-omni`，2026-07-27）

## 结论摘要

1. 当前项目的 `app/tts_qwen.py` 是直接调用官方 `qwen-tts` Python wrapper 的单实例、单请求路径；它不是 vLLM-Omni 的 continuous batching 实现。
2. 官方 `qwen-tts` 的 `generate_custom_voice()` 支持把多个文本作为一个显式 list batch 传入，但当前官方 Python wrapper 在 `stream_pcm()` 这类调用中返回完整 waveform 后才解码，不能提供真正的增量音频生成。官方源码还明确写明 `non_streaming_mode=False` 目前只是模拟 streaming text input，不会启用真正的 streaming generation。
3. vLLM-Omni 当前的 Qwen3-TTS 在线服务是另一套实现：Talker stage 使用 vLLM scheduler，Code2Wav stage 通过 Omni connector 传递 codec chunk；stage worker 可让多个请求进入同一个 stage batch。其官方文档建议通过 `--stage-overrides` 调整 `max_num_seqs`，而不是在应用层串行调用 Python model。
4. Qwen 官方 README（该上游 commit）仍写着 vLLM-Omni 对 Qwen3-TTS 只支持 offline inference、online serving later；但 vLLM-Omni 当前 HEAD 已有在线 speech API 和 Qwen3-TTS online examples。这是两边文档不同步，不能把 Qwen README 的旧状态当作当前 vLLM-Omni 能力边界。
5. vLLM-Omni 最新在线 Qwen3-TTS 文档已明确列出 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 和 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 为支持的更小/更快变体。因此，0.6B CustomVoice 是 vLLM-Omni 官方声明支持的模型；但 upstream 当前公开的 `run_server.sh` 和在线 E2E 测试仍固定 1.7B CustomVoice，未提供 0.6B 的同等 CI 覆盖。生产切换前仍须在目标镜像、GPU 与本地 checkpoint 上做专项验收。

## Qwen3-TTS 官方 Python 实现

### 模型加载

官方 README 的 CustomVoice 示例直接使用：

```python
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

来源：

- [Qwen3-TTS README](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md#L148-L162)
- [官方 CustomVoice 示例](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/examples/test_model_12hz_custom_voice.py#L23-L32)

wrapper 的 `from_pretrained()` 将收到的 kwargs 原样转发给 `AutoModel.from_pretrained()`，然后再独立加载 `AutoProcessor`；它没有官方定义的 `device_map="auto"` 或 CPU fallback。见：

- [qwen3_tts_model.py](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py#L82-L121)

因此，当前项目为了捕获 `Cannot copy out of meta tensor` 而在 `app/tts_qwen.py` 中尝试 `device_map="auto"`、再 CPU 加载后 `.to(cuda)`，属于项目自有兼容性 workaround，不是 Qwen 官方推荐流程。它可能掩盖真正的运行时兼容性问题（Torch/Transformers/Accelerate/权重格式或 checkpoint 配置），不能替代在官方依赖组合下验证原始 `device_map="cuda:0"` 加载。

### CustomVoice 与 batch

Qwen README 说明 CustomVoice 的参数是 `text`、`language`、`speaker` 和可选 `instruct`，并展示了单请求和 list batch 两种调用。来源：[README](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md#L148-L184)。

官方 wrapper 的实现会把 scalar 归一化成 list，检查四个列表长度一致，随后一次调用底层 `self.model.generate(...)`，最后一次性调用 speech tokenizer 解码完整 waveform：

- [generate_custom_voice() 参数和 batch 校验](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py#L732-L839)

其中源码文档明确写出：

> `non_streaming_mode=False` currently only simulates streaming text input, rather than enabling true streaming input or streaming generation.

来源：[qwen3_tts_model.py](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py#L738-L755)。

Qwen README 虽然宣称模型具备低延迟 streaming generation，但 Python wrapper 的 `generate_custom_voice()` 返回类型仍是 `(List[np.ndarray], sample_rate)`，并在返回前完成完整 decode；因此“模型架构支持 streaming”和“当前 Python API 可向 HTTP 客户端逐 chunk 输出”是两个不同层次。

## vLLM-Omni 官方实现

### 在线接口和部署命令

vLLM-Omni 当前提供 Qwen3-TTS 的 online serving 示例，命令为：

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091
```

并通过 `vllm_omni/deploy/qwen3_tts.yaml` 配置 stage pipeline。来源：[online serving text-to-speech 文档](https://github.com/vllm-project/vllm-omni/blob/8001bb155dae5798a1ae891ae2529a314c6ee99a/docs/user_guide/examples/online_serving/text_to_speech.md#L307-L341) 和 [run_server.sh](https://github.com/vllm-project/vllm-omni/blob/8001bb155dae5798a1ae891ae2529a314c6ee99a/examples/online_serving/text_to_speech/qwen3_tts/run_server.sh#L31-L39)。

当前 deploy config 的核心是 `async_chunk: true`、stage 0 Talker、stage 1 Code2Wav，以及 shared-memory connector：

- [qwen3_tts.yaml](https://github.com/vllm-project/vllm-omni/blob/8001bb155dae5798a1ae891ae2529a314c6ee99a/vllm_omni/deploy/qwen3_tts.yaml#L1-L98)

### Continuous batching 的实际边界

vLLM-Omni 的 speech API 文档说明 batch endpoint 会把 items 并发提交给 `generate()`，stage worker 按配置的 `max_batch_size` 自动 batch，超出的请求排队；客户端不需要自己节流。文档还建议为提高吞吐把两个 stage 的 `max_num_seqs` 设为大于 1，并给出 Qwen3-TTS CustomVoice 的 `max_num_seqs: 10` 示例：

- [speech_api.md batch 配置](https://github.com/vllm-project/vllm-omni/blob/8001bb155dae5798a1ae891ae2529a314c6ee99a/docs/serving/speech_api.md#L650-L669)
- [batch_speech_client.py](https://github.com/vllm-project/vllm-omni/blob/8001bb155dae5798a1ae891ae2529a314c6ee99a/examples/online_serving/text_to_speech/qwen3_tts/batch_speech_client.py#L1-L27)

Qwen3-TTS 专门的 tuning 文档进一步说明 stage 0/Talker 和 stage 1/Talker's downstream path 使用 continuous batching，而 Code2Wav 采用 static batching；因此不能笼统地说整个两阶段 pipeline 都是同一种 continuous batching：

- [Qwen3 Omni/TTS 性能优化设计](https://github.com/vllm-project/vllm-omni/blob/8001bb155dae5798a1ae891ae2529a314c6ee99a/docs/design/qwen3_omni_tts_performance_optimization.md#L108-L124)

### 支持的模型版本

vLLM-Omni 最新的 Qwen3-TTS 在线文档明确列出：

- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

其中 0.6B CustomVoice 的说明是“更小/更快的变体”。文档还明确说明 `stream=true` 且 `response_format="pcm"` 会在 Code2Wav 解码中输出原始 PCM chunk。来源：[vLLM-Omni Qwen3-TTS online serving 文档](https://docs.vllm.com.cn/projects/vllm-omni/en/latest/user_guide/examples/online_serving/qwen3_tts/)。

需要区分“官方支持”和“公开 CI 覆盖”：当前 upstream `run_server.sh` 及 online E2E `test_qwen3_tts_customvoice.py` 仍使用 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`。这不是对 0.6B 的否定，但说明 0.6B 不能直接继承 1.7B 的性能和稳定性结论。来源：[run_server.sh](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/text_to_speech/qwen3_tts/run_server.sh) 和 [online E2E test](https://github.com/vllm-project/vllm-omni/blob/main/tests/e2e/online_serving/test_qwen3_tts_customvoice.py)。

## 当前项目逐项核对

| 项目 | 当前实现 | 官方/主流实现 | 判断 |
|---|---|---|---|
| 模型加载 | `Qwen3TTSModel.from_pretrained()`；先用 `tts_device`，遇到 meta tensor 再试 `auto`/CPU | Qwen 示例直接 `device_map="cuda:0"` + BF16 + 可选 FlashAttention 2 | API 形态正确，但 fallback 是项目特有 workaround；应先固定并验证官方依赖/加载路径 |
| 请求并发 | `Qwen3TTSSynthesizer` 只有一个 `_model`，`stream_pcm()` 每次调用一次 `generate_custom_voice()` | vLLM-Omni stage scheduler + `max_num_seqs`/`max_batch_size`，请求可同时进入 stage batch | 当前没有 continuous batching |
| batch | 每个 HTTP 请求只传一个 `text`，没有把多个请求合并为 list batch | 官方 qwen-tts 支持显式 list batch；vLLM-Omni 提供 `/v1/audio/speech/batch` | 当前没有利用 batch 能力 |
| 流式输出 | 当前 `stream_pcm()` 在 `generate_custom_voice()` 返回完整 waveform 后只 yield 一次 PCM | vLLM-Omni `async_chunk` 通过 connector 输出 codec/audio chunks | 当前接口是“伪流式”（响应封装为 stream，但首 chunk 需等待完整生成） |
| 0.6B | Qwen 本地 Python wrapper 可按官方 CustomVoice API 使用 | vLLM-Omni 最新在线文档明确列出 0.6B CustomVoice；公开启动脚本/E2E 仍固定 1.7B | 可作为 vLLM-Omni 接入目标，但必须专项验证 |
| 采样参数 | 当前项目只传 `text/language/speaker/instruct`，使用 qwen-tts 默认 generate 参数 | 官方示例允许 `max_new_tokens` 等 HF `generate` kwargs；vLLM deploy config 显式设 `min_tokens: 2` 等默认值 | 当前没显式设置 `min_new_tokens/min_tokens`；空音频或 EOS 首 token 风险应由线上日志验证 |

## 对当前故障的解释

`Cannot copy out of meta tensor` 发生在 HuggingFace/Accelerate 模型加载阶段，不是 continuous batching 调度问题。当前项目的 fallback 只能改变加载尝试顺序；它不会让 qwen-tts Python path 变成 vLLM-Omni，也不会带来请求级 batch 或真正的增量音频。

更符合官方实现的两条路线是：

1. **保留 qwen-tts Python backend（0.6B 可行性优先）**：固定官方示例的依赖组合、固定 `device_map="cuda:0"`、在容器启动时做一次显式 model-load smoke test；并接受该路径是单进程/单请求生成，需在应用层另行设计受控并发或显式 list batch。
2. **采用 vLLM-Omni online serving（continuous batching/低 TTFA 优先）**：使用 vLLM-Omni 的 Qwen3-TTS deploy config 和 `/v1/audio/speech` API，以 `stream=true` 和 `response_format="pcm"` 获取真正的增量音频。0.6B CustomVoice 已被最新在线文档列为支持模型，但应在当前目标硬件做专项性能、流式正确性和稳定性验收；不能直接把当前 Python adapter 的 `stream_pcm()` 改名为 continuous batching。
