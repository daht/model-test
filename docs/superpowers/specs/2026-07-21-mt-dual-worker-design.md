# MT 单卡双 Worker 设计

## 目标

在现有单张 A10、单个 `hy-mt-api` Docker Compose 容器内启动两个 Uvicorn Worker，使两个独立 MT 模型实例能够并行处理请求，并保持现有 `8000` 端口和 HTTP API 不变。

## 架构

`Dockerfile` 的启动命令固定增加 `--workers 2`。Uvicorn 父进程监听容器内的 `8000` 端口，并将连接分发给两个 Worker。每个 Worker 独立导入 `app.main`、创建 `TransformersTranslator`、延迟加载一份模型，并保留自己的进程内推理锁。

不增加 Nginx、HAProxy、Kubernetes Service、第二个容器或新的运行时配置。Docker Compose 的端口、健康检查、模型只读挂载和 GPU 挂载保持不变，现有监控脚本仍只监控一个 `hy-mt-api` 容器。

## 请求与故障行为

请求继续访问同一个 `/v1/translate` endpoint。单个 Worker 内的请求仍串行执行，但两个 Worker 可以各执行一路推理。一个 Worker 正在执行长请求时，另一个 Worker 仍能接受请求。

两个 Worker 都使用相同 API 密钥、模型目录和配置。任一 Worker 异常退出时由 Uvicorn 父进程负责管理；容器级健康检查和 Docker 重启策略保持现状。本次不新增应用层重试，避免翻译请求被重复执行。

## 资源与风险

每个 Worker 独立加载模型，预计 MT 模型显存从约 `4,684 MiB` 增加到约 `9,368 MiB`。结合本轮监控记录的其他 GPU 进程约 `1,262 MiB`，总量仍低于 A10 的 `23,028 MiB`，但最终容量必须通过双 Worker 实测确认。

首次流量可能分别触发两个 Worker 加载模型，因此发布后必须完成足够次数的预热，再进行正式压测。双 Worker 可能提高 GPU 利用率和吞吐，也可能因 GPU 计算竞争增加单请求延迟；不预先假设吞吐能够翻倍。

## 验证

- 增加部署结构测试，断言 MT Dockerfile 使用且仅使用 `--workers 2`。
- 保持 ASR 的单 Worker 约束不变。
- 运行 MT 聚焦测试和仓库提交门禁。
- 云端重新构建并重启 `hy-mt-api` 后，确认容器健康、两个 Worker 均加载模型、无 OOM，然后使用相同语料和并发档位复测。

## 非目标

本次不删除 `TransformersTranslator` 的进程内锁，不实现动态批处理，不修改翻译 API，不改变 ASR/TTS 服务，也不新增独立网关。
