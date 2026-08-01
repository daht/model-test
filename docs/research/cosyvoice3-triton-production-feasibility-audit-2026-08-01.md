# CosyVoice3 Triton 实时高并发可行性审计（2026-08-01）

## 范围与结论

审计对象为官方 CosyVoice commit
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` 的
`runtime/triton_trtllm`，以及本机 A10 的 C=1 归档
`log/20260801T132131Z-b2d34c.tar.gz`。

**结论：当前官方 runtime 不能通过配置达到高并发、连续播放的实时 TTS 要求。**
它的下游 `token2wav` 和 vocoder 没有跨请求 GPU batch；更关键的是
CosyVoice3 Flow 的推理入口显式限制业务 batch 为 1。要改变这一点需要维护
CosyVoice fork、重新导出/构建 TensorRT engine、实现变长批处理和按 deadline 的流调度，
不是 Triton 参数调优。

这条路线有理论上的改造空间，但它是新的推理后端项目，当前没有足以承诺 A10
实时并发目标的实现或性能证据。

## 本机运行证据

本轮公网 WS C=1 预检为 3/3 成功、TTFA p95 `756 ms`，但长文本 chunk gap p99/max
为 `1.499 s`，出现 1 次、共 `2.691 s` 播放断流。因此在 C=1 已失败，不能进入 C=2。

归档中 323 份 Triton metrics 快照显示：

| 模型 | successful requests | executions | average batch size |
| --- | ---: | ---: | ---: |
| `token2wav` | 87 | 87 | 1.00 |
| `vocoder` | 87 | 87 | 1.00 |

两个下游阶段累计 queue time 分别只有 `4.587 ms` 和 `3.632 ms`，不是排队造成
1.499 秒间隔；累计 compute time 分别是 `30.773 s`、`10.754 s`。这与逐 chunk
串行下游计算一致。

## 官方实现限制

### 1. 运行脚本明确设置单请求、零等待

`run_cosyvoice3.sh` 在生成模型仓库时固定：

```bash
MAX_QUEUE_DELAY_MICROSECONDS=0
TRITON_MAX_BATCH_SIZE=1
```

然后填入顶层、`token2wav` 和 vocoder 的 `config.pbtxt`。

来源：[run_cosyvoice3.sh](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/run_cosyvoice3.sh#L64-L95)。

调大这些两个数不会实现 GPU batch：Triton 最多只会把多个 request 传给 Python
backend，backend 本身仍必须组装一次 B>1 forward。

### 2. 当前 Python backend 显式逐请求调用

`token2wav/1/model.py` 遍历 `requests`，对每一项单独补出 batch 维，构造一个长度为 1
的 `token_len`，并单独调用 `self.flow.inference(...)`。vocoder 也逐项补维并调用一次
`self.hift.inference(...)`。它们没有 padding、length/mask、stack 或 batched-output
拆分逻辑。

来源：

- [token2wav backend](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/model_repo_cosyvoice3/token2wav/1/model.py#L134-L200)
- [vocoder backend](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/model_repo_cosyvoice3/vocoder/1/model.py#L47-L69)

### 3. CosyVoice3 Flow 本身拒绝业务 B>1

`CausalMaskedDiffWithDiT.inference()` 有：

```python
assert token.shape[0] == 1
```

并为单项形状创建条件和 mask（例如 `torch.zeros([1, ...])`）。官方同一仓库的
`token2wav_cosyvoice3.py` 已明确说明该断言，因此 offline 路径也逐样本调用 Flow。

来源：

- [Flow inference](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/flow/flow.py#L235-L281)
- [CosyVoice3 offline wrapper](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/token2wav_cosyvoice3.py#L263-L302)

因此仅将 Triton `max_batch_size` 改为 2 或更大，最终会进入逐项循环或触发 Flow
断言；它不会将两个用户流变成一次安全的 Flow forward。

### 4. TensorRT engine 也不是可直接扩大的用户 batch

官方 `token2wav` 的 TensorRT optimization profile 使用固定的首维 `2`，并在
`forward_estimator()` 中把引擎输入 shape 强制设为 `(2, 80, T)`。该 `2` 来自
扩散/CFG 内部计算，不代表两条 TTS 会话。将外层业务 batch 改为 B，需要设计内部
shape 规则、修改 Flow 实现、重新导出 ONNX 并重建 profile/engine。

来源：

- [TRT profile](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/model_repo_cosyvoice3/token2wav/1/model.py#L109-L132)
- [TRT execution shape](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/flow/flow_matching.py#L126-L153)

### 5. vocoder 有 batch 维，但没有变长流式批接口

`CausalHiFTGenerator.inference()` 按 tensor 的首维计算，因此理论上可以接受等长 B>1
mel。然而现有 backend 不传 length/mask，流式 BLS 又为每条会话维护不同的累计 mel
长度和 `speech_offset`。直接 padding 后 forward 会引入尾部和 cache/切片正确性问题；
必须实现长度分桶、输出裁剪，以及每流 finalize/cancel 状态验证后才可使用。

来源：[CausalHiFTGenerator inference](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/hifigan/generator.py#L713-L726)。

## 若继续投入，最小正确改造单元

1. 修改并单测 Flow，使 B>1 的 token、prompt token、prompt mel、embedding、length 和
   mask 都正确 padding/构造；先做离线等长 B=2 数值与音频回归。
2. 按新的业务 B 重新导出 ONNX 与 TensorRT profile，验证没有把 CFG 内部首维和用户 batch
   混淆。
3. 为 vocoder 实现等长/长度分桶的 B>1 forward、每项音频裁剪及流式音频一致性测试。
4. 在 BLS 前放置 bounded deadline-aware batcher：只在小窗口内合并同 deadline bucket 的
   chunk，buffer 即将耗尽时立即执行；每条流保留独立 offset、finalize 和取消状态。
5. 以 C=1 零断流为门槛，再做 C=2/4 阶梯压测，并要求 `token2wav` 与 vocoder 的 metrics
   同时出现稳定 `average batch size > 1`。

第 1 步之前，没有一个可信的运行时参数实验可改善本轮 1.499 秒单流 gap。

## 决策建议

不要将该官方 runtime 作为近期高并发实时上线候选。若业务必须使用 CosyVoice3，可将上述
五步作为受限的研发 POC；其第一关是“离线 Flow B=2 保持音频等价”，失败即停止。
若近期目标是稳定上线，应恢复并横向扩展已验证的 Qwen 单流路径，而不是继续在该 Triton
配置上试探参数。
