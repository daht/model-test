# ASR + MT 商业成本总报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于已验证的 ASR、MT 容量证据，生成一份百万用户规模的 ASR + MT 商业成本总报告，并提供内容一致的 Markdown 和 HTML 版本。

**Architecture:** 报告以现有容量报告和原始压测产物为唯一事实来源，先建立统一业务量、70% 规划、分服务 N+1 和 A10 单价口径，再分别计算 ASR、MT 与合计成本。Markdown 是内容源，HTML 使用现有成本报告的响应式卡片与表格视觉，但不得改变数字、证据状态或适用边界。

**Tech Stack:** Markdown、静态 HTML/CSS、Python 标准库校验、现有 pytest 回归测试。

---

### Task 1: 建立证据与计算基线

**Files:**
- Read: `docs/asr-three-model-a10-comparison-2026-07-19.md`
- Read: `docs/asr-sensevoice-a10-capacity-evaluation-2026-07-19.md`
- Read: `docs/mt-vllm-a10-capacity-evaluation-2026-07-22.md`
- Read: `log/mt-vllm-soak-c48-20260722T020625Z/mt-benchmark.json`
- Read: `docs/million-user-cost-report.html`

- [ ] **Step 1: 核对 ASR 容量状态**

确认三种 ASR 的“当前有余量建议”分别为 SenseVoice Small 80 路、Qwen3-ASR-0.6B 24 路、Qwen3-ASR-1.7B 12 路，并保留各自测试时长、实时 lag 和实现差异说明。

- [ ] **Step 2: 核对 MT 一小时原始结果**

从 `mt-benchmark.json` 读取并确认：并发 48、成功 315,092/315,105、吞吐 5,727.416050662647 源字符/秒、P95 0.7455768829677254 秒、错误率 4.1256089240094575e-05、月容量 14,845,462,403.317581 字符。

- [ ] **Step 3: 使用原始精度复算全部成本**

Run:

```bash
python3 - <<'PY'
from math import ceil

monthly_gpu = 2132.72
month_seconds = 30 * 24 * 3600
audio_seconds = 438_082_143
translation_characters = 2_142_857_143
average_streams = audio_seconds / month_seconds

plans = {
    "SenseVoice Small": 80,
    "Qwen3-ASR-0.6B": 24,
    "Qwen3-ASR-1.7B": 12,
}

for name, recommended in plans.items():
    asr_business = ceil(average_streams / (recommended * 0.70))
    asr_n1 = asr_business + 1
    total_cards = asr_n1 + 2
    print(name, asr_business, asr_n1, total_cards,
          f"{total_cards * monthly_gpu:,.2f}",
          f"{total_cards * monthly_gpu * 12:,.2f}")

assert round(average_streams, 2) == 169.01
assert ceil(average_streams / 56) == 4
assert 2_142_857_143 < 10_391_823_682
assert round(7 * monthly_gpu, 2) == 14_929.04
assert round(7 * monthly_gpu * 12, 2) == 179_148.48
PY
```

Expected:

```text
SenseVoice Small 4 5 7 14,929.04 179,148.48
Qwen3-ASR-0.6B 11 12 14 29,858.08 358,296.96
Qwen3-ASR-1.7B 21 22 24 51,185.28 614,223.36
```

### Task 2: 编写 Markdown 总报告

**Files:**
- Create: `docs/asr-mt-commercial-cost-report-2026-07-22.md`

- [ ] **Step 1: 写执行摘要与主结论**

报告开头明确：推荐 SenseVoice Small + Hy-MT2/vLLM；ASR 5 张、MT 2 张、合计 7 张 A10；纯 GPU 14,929.04 元/月、179,148.48 元/年。明确这是百万用户线性外推、70% 规划和分服务 N+1 口径。

- [ ] **Step 2: 写业务量与公式**

列出 438,082,143 音频秒/月、169.01 平均持续路数、2,142,857,143 翻译字符/月，以及以下公式：

```text
ASR 业务卡 = ceil(月音频秒 ÷ 2,592,000 ÷ 推荐并发 ÷ 70%)
ASR 部署卡 = ASR 业务卡 + 1
MT 业务卡 = ceil(月源字符量 ÷ 10,391,823,682)
MT 部署卡 = MT 业务卡 + 1
```

- [ ] **Step 3: 写推荐部署与成本明细**

分别给出 SenseVoice Small 的 88 路已验证上界、80 路有余量建议、56 路规划容量和 5 张部署；给出 MT 并发 48 一小时证据、103.92 亿字符规划容量和 2 张部署。明确 ASR 与 MT 使用独立 GPU 池，不把显存账面余量当作混部证据。

- [ ] **Step 4: 写 ASR 替代方案表**

加入三行总成本表：SenseVoice 7 张/14,929.04 元月成本；Qwen 0.6B 14 张/29,858.08 元；Qwen 1.7B 24 张/51,185.28 元。说明三个容量数字测试口径不同，模型选择还需要统一 CER/WER 和真实业务质量验收。

- [ ] **Step 5: 写市场参考、风险与证据状态**

市场参考只引用带日期的官方公开价格并注明价格会变化。风险部分明确：SenseVoice 未完成长时 soak；ASR 三模型未完成统一质量门禁；MT 已完成一小时稳定性但没有 24 小时、多实例切换和真实业务质量验证；成本不包含非 GPU 项目。

- [ ] **Step 6: 扫描报告边界**

Run:

```bash
rg -n -i 'api[_-]?key\s*[=:]|authorization:\s*bearer|https?://([0-9]{1,3}\.){3}[0-9]{1,3}' docs/asr-mt-commercial-cost-report-2026-07-22.md
```

Expected: no output.

### Task 3: 生成 HTML 阅读版

**Files:**
- Create: `docs/asr-mt-commercial-cost-report-2026-07-22.html`
- Reference: `docs/million-user-cost-report.html`
- Reference: `docs/mt-a10-dual-worker-capacity-evaluation-2026-07-21.html`

- [ ] **Step 1: 复用现有报告视觉语言**

使用单文件 HTML/CSS，包含标题区、摘要 KPI 卡片、部署数量卡片、成本表、方案对比表、风险提示和证据区。支持 820px 与 480px 窄屏断点、横向表格滚动和 A4 打印样式；不引入 JavaScript、外部字体或第三方资源。

- [ ] **Step 2: 保持 Markdown 与 HTML 数字一致**

HTML 必须包含以下字符串：

```text
7 张 A10
14,929.04 元/月
179,148.48 元/年
SenseVoice Small
Hy-MT2-1.8B + vLLM
29,858.08 元/月
51,185.28 元/月
```

- [ ] **Step 3: 校验 HTML 结构与敏感信息**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
import re

path = Path("docs/asr-mt-commercial-cost-report-2026-07-22.html")
text = path.read_text(encoding="utf-8")

class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        if tag not in {"meta", "link", "img", "br", "hr", "input", "source"}:
            self.stack.append(tag)
        for key, value in attrs:
            if key == "id" and value:
                assert value not in self.ids, value
                self.ids.add(value)
    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1] == tag, (tag, self.stack[-5:])
        self.stack.pop()

parser = Parser()
parser.feed(text)
parser.close()
assert not parser.stack

for required in (
    "7 张 A10", "14,929.04 元/月", "179,148.48 元/年",
    "SenseVoice Small", "Hy-MT2-1.8B + vLLM",
    "29,858.08 元/月", "51,185.28 元/月",
):
    assert required in text, required

for pattern in (
    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
    r"(?i)bearer\s+[A-Za-z0-9._~-]{8,}",
):
    assert not re.search(pattern, text), pattern

print("HTML structure, required values, and sensitive-value scan: ok")
PY
```

Expected: `HTML structure, required values, and sensitive-value scan: ok`.

### Task 4: 增加入口并完成一致性验证

**Files:**
- Modify: `README.md`
- Verify: `docs/asr-mt-commercial-cost-report-2026-07-22.md`
- Verify: `docs/asr-mt-commercial-cost-report-2026-07-22.html`

- [ ] **Step 1: 在 README 增加总报告入口**

在现有文档列表中加入 Markdown 与 HTML 两个路径，描述为百万用户 ASR + MT 容量、部署数量和纯 GPU 成本总报告。

- [ ] **Step 2: 校验本地 Markdown 链接**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

files = [Path("README.md"), Path("docs/asr-mt-commercial-cost-report-2026-07-22.md")]
broken = []
for source in files:
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (source.parent / target.split("#", 1)[0]).resolve()
        if not resolved.exists():
            broken.append((str(source), target))
assert not broken, broken
print("local Markdown links: ok")
PY
```

Expected: `local Markdown links: ok`.

- [ ] **Step 3: 运行相关回归测试**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_benchmark_mt.py \
  tests/test_monitor_mt_benchmark.py \
  tests/test_mt_vllm_deployment.py \
  tests/test_monitor_asr_bottleneck.py
```

Expected: all selected tests pass.

- [ ] **Step 4: 检查格式和变更边界**

Run:

```bash
git diff --check
git status --short -- \
  README.md \
  docs/asr-mt-commercial-cost-report-2026-07-22.md \
  docs/asr-mt-commercial-cost-report-2026-07-22.html
```

Expected: `git diff --check` exits 0；状态只显示计划内文件，不包含音频、日志、模型、密钥或缓存。

- [ ] **Step 5: 提交候选报告**

```bash
git add -- \
  README.md \
  docs/asr-mt-commercial-cost-report-2026-07-22.md \
  docs/asr-mt-commercial-cost-report-2026-07-22.html
scripts/verify_asr_release.sh commit
git commit -m "docs: add combined ASR and MT cost report"
```

Expected: commit gate passes while intended files are staged, then the report candidate is committed without unrelated files.
