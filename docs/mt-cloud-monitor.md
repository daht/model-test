# MT 云端压测监控

`scripts/monitor_mt_benchmark.sh` 在云服务器上同时观察 `hy-mt-api` 网关和 `hy-mt-vllm` 推理容器。它不会发送翻译请求，不会读取压测地址或密钥，也不会启动、重启、构建或修改服务。压测流量仍由另一台机器上的 `scripts/benchmark_mt.py` 产生。

## 前提

- 在包含当前 Docker Compose 项目的仓库目录中执行。
- Compose 服务默认名为 `hy-mt-api` 和 `hy-mt-vllm`，两者都必须已经运行，并且各自恰好有一个容器。
- 主机已安装 Bash、Docker Compose v2、NVIDIA驱动工具、Python 3、tar和sha256sum。
- 开始前确认应用的服务日志不会输出请求正文、译文或密钥。监控器会原样保存压测窗口内的服务日志，无法可靠清洗应用自行输出的敏感内容。

## 启动

在云服务器执行：

```bash
scripts/monitor_mt_benchmark.sh
```

看到 `MT benchmark monitor started.` 后，在另一台机器启动MT压测。压测结束后回到云服务器终端按 `Ctrl+C`。监控器停止自身采集进程并自动生成报告、校验清单和压缩包；它不会影响翻译服务进程。

## 默认配置

- 网关服务：`hy-mt-api`
- vLLM服务：`hy-mt-vllm`
- GPU：`0`
- 输出根目录：`/tmp/mt-monitor`
- GPU和GPU进程采样：每`0.5`秒
- 容器采样：每`1`秒
- 保留最近`20`次已完成运行
- 最长保留`14`天

可以在启动前设置以下环境变量覆盖默认值：

- `MT_MONITOR_GATEWAY_SERVICE`
- `MT_MONITOR_VLLM_SERVICE`
- `MT_MONITOR_GPU_INDEX`
- `MT_MONITOR_OUTPUT_ROOT`
- `MT_MONITOR_GPU_INTERVAL_SECONDS`
- `MT_MONITOR_CONTAINER_INTERVAL_SECONDS`
- `MT_MONITOR_KEEP_RUNS`
- `MT_MONITOR_KEEP_DAYS`

这些变量只描述本机监控配置，不接受服务地址、API密钥或压测语料。

## 产物

每次运行写入：

```text
/tmp/mt-monitor/runs/<run-id>/
```

其中包含：

- `metadata.json`
- `gpu.csv`
- `gpu-processes.csv`
- `gateway-container.csv`
- `vllm-container.csv`
- `gateway-service.log`
- `vllm-service.log`
- `collector-errors.log`
- `report.json`
- `report.md`
- `manifest.sha256`
- `.completed`

同级目录还会生成 `<run-id>.tar.gz`。把该压缩包交给分析人员即可。不要只提供 `report.md`，原始采样和校验清单也是容量判断证据。

## 结果边界

云端监控报告只解释GPU、GPU进程、容器和服务状态。RPS、延迟、Token吞吐、每百万字成本及最高可持续并发仍以本地MT压测生成的 `mt-benchmark.json` 和 `mt-benchmark.md` 为准。两组数据通过UTC时间窗口对应。
