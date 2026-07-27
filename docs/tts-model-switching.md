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

脚本检测到 `TTS_BACKEND=qwen` 时不会要求 Triton 容器，并会安装
`requirements-tts-qwen.txt`。Qwen Python 包当前的 `non_streaming_mode=False`
仍是模拟增量文本输入，不能视为真正的增量音频生成；测试报告必须标注这一差异。

两种后端都使用同一个 `/v1/tts` 和 `/v1/tts/stream` 协议，压测脚本只需要替换
`TTS_MODEL_NAME` 和输出目录。
