# TTS 部署级模型切换

TTS adapter 每个部署实例只加载一个模型。通过 `.env` 选择后端，重建 adapter
即可切换；请求不会在多个模型之间动态路由，也不会在同一进程中同时占用两套模型显存。

## CosyVoice3 Triton

```dotenv
TTS_BACKEND=triton
TTS_MODEL_NAME=Fun-CosyVoice3-0.5B-2512
TTS_MODEL_ID=/models/CosyVoice
TTS_TRITON_URL=127.0.0.1:18001
TTS_TRITON_MODEL_NAME=cosyvoice3
TTS_SAMPLE_RATE=24000
```

启动 Triton 后运行：

```bash
scripts/run_tts_triton_adapter.sh --foreground
```

## Qwen3-TTS 0.6B

Qwen 的 0.6B CustomVoice 模型 ID 为：
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`。模型可以改为本地目录。

```dotenv
TTS_BACKEND=qwen
TTS_MODEL_NAME=Qwen3-TTS-0.6B
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_DEVICE=cuda:0
TORCH_DTYPE=bfloat16
TTS_QWEN_SPEAKER=Vivian
TTS_QWEN_LANGUAGE=Chinese
TTS_QWEN_INSTRUCT=
TTS_SAMPLE_RATE=24000
```

然后重建 adapter：

```bash
docker rm -f cosyvoice-tts-api 2>/dev/null || true
scripts/run_tts_triton_adapter.sh --foreground
```

脚本首次运行时会自动构建 `cosyvoice-tts-adapter:latest` 镜像，将 adapter 和
Qwen 依赖预装进去；后续重建容器不会再次执行 pip 安装。也可以通过
`TTS_API_IMAGE` 指定已经构建好的兼容镜像。脚本检测到 `TTS_BACKEND=qwen` 时不会要求
Triton 容器。Qwen Python 包当前的 `non_streaming_mode=False`
仍是模拟增量文本输入，不能视为真正的增量音频生成；测试报告必须标注这一差异。

## Qwen3-TTS 0.6B vLLM-Omni

该后端不在 adapter 容器中加载 Qwen 或占用 GPU。它将请求转发到独立的
vLLM-Omni `/v1/audio/speech` 服务，并使用 `stream=true` 和
`response_format=pcm` 接收真实的增量 PCM chunk。

```dotenv
TTS_BACKEND=vllm_omni
TTS_MODEL_NAME=Qwen3-TTS-0.6B
TTS_VLLM_OMNI_BASE_URL=http://127.0.0.1:8091
TTS_VLLM_OMNI_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_VLLM_OMNI_TIMEOUT_SECONDS=300
TTS_QWEN_SPEAKER=Vivian
TTS_QWEN_LANGUAGE=Chinese
TTS_QWEN_INSTRUCT=
TTS_SAMPLE_RATE=24000
```

如果 vLLM-Omni 配置了 API key，额外设置 `TTS_VLLM_OMNI_API_KEY`。adapter 在启动时
检查上游 `/health`；上游不可用时 adapter 不会报告健康。外部接口仍为 `/v1/tts` 和
`/v1/tts/stream`，其中 WebSocket 会在收到每个上游 PCM chunk 后立即转发。

三种后端都使用同一个 `/v1/tts` 和 `/v1/tts/stream` 协议，压测脚本只需要替换
`TTS_MODEL_NAME` 和输出目录。

配置 `TTS_BACKEND=vllm_omni` 后，直接运行
`scripts/run_tts_triton_adapter.sh --foreground` 即可一键启动或复用 vLLM-Omni，
等待其健康后再启动 adapter。
