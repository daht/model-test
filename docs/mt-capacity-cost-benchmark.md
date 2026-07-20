# MT 容量与每百万源字符成本压测

`scripts/benchmark_mt.py` 从本机发起真实 HTTP 请求，经过翻译服务的鉴权、FastAPI、排队和模型推理链路。它用于确定独立单张 A10 在指定延迟与错误率门槛下的可持续容量，并计算每 1,000,000 个源文本 Unicode 字符的 GPU 成本。

## 前提

- 目标服务运行真实 HY-MT 模型，不能使用 `mock` 后端作为商业容量证据。
- A10 由 MT 独占；与 ASR/TTS 共用 GPU 的结果不能作为本报告的单卡容量。
- 模型已加载并预热。
- 准备真实业务语料和对应模型 tokenizer。
- endpoint 和密钥只通过当前 shell 环境提供，不写进文档或报告。

## 准备语料

语料为 UTF-8 JSONL，每行只包含三个字段：

```json
{"source_lang":"zh","target_lang":"en","text":"需要翻译的业务文本"}
```

程序按 `text` 的 Unicode 字符数计费，包含中文、英文、标点和空格，不做归一化。压测产物只保存记录数、字符数和语言方向汇总，不保存文本或译文。

## 安全配置

在执行进程的环境中设置：

```bash
export MT_BENCHMARK_URL="<configured-by-operator>"
export API_KEY="<configured-by-operator>"
```

不要把真实值放入脚本、命令参数、文档、聊天记录或压测产物。报告和正常终端输出不会显示这两个值。

## 执行

下面的命令不包含 endpoint 或密钥：

```bash
python scripts/benchmark_mt.py \
  --corpus /secure/input/mt-corpus.jsonl \
  --tokenizer /models/HY-MT1.5-1.8B \
  --output-dir /secure/output/mt-benchmark \
  --concurrency 1,2,4,8,16,32 \
  --duration-seconds 30 \
  --max-p95-seconds 1.0 \
  --max-error-rate 0.001
```

默认参数：

- 预热请求：3
- 请求超时：30 秒
- A10 月成本：`2132.72` 元
- 月运行时间：30 天，即 2,592,000 秒

提高正式压测的单档持续时间时，使用 `--duration-seconds`；覆盖单卡月价时，使用 `--a10-monthly-cost-cny`。

## 输出与指标

输出目录包含：

- `mt-benchmark.json`：机器可读结果。
- `mt-benchmark.md`：可直接查看的汇总。

每个并发档记录：

- 请求数、成功数、失败数、RPS和错误分类；
- 源字符/秒、输入 Token/秒、输出 Token/秒；
- 延迟最小值、平均值、P50、P95、P99和最大值；
- 月处理源字符容量；
- 每百万源字符 GPU 成本；
- 是否同时满足P95和错误率门槛。

最高可持续档是配置列表中满足全部门槛的最高并发，并不等于刚好不OOM的极限并发。

## 成本公式

```text
月处理源字符 = 实测成功源字符/秒 × 2,592,000

每百万源字符 GPU 成本
= 2,132.72 × 1,000,000 ÷ 月处理源字符
```

该数字仅包含单张A10 GPU成本，不包含CPU、内存、磁盘、网络、负载均衡、故障冗余和运维。商业报价应在GPU成本之外另行加入这些项目。

## 验收解释

本地测试可以验证脚本行为，但不能建立商业容量结论。正式结果必须来自：真实HY-MT模型、独立A10、真实业务语料、稳定运行窗口和目标SLO。若没有任何并发档通过，命令仍会生成报告并明确记录“无可持续档”，不得挑选失败档作为容量依据。
