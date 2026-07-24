# 商业 TTS API 输出采样率与音频格式审计

查询日期：**2026-07-23**

范围：仅使用腾讯云、火山引擎/豆包、MiniMax 的官方产品文档、官方 API 文档或官方 SDK/示例源码。本文讨论语音合成输出，不把 ASR 输入格式、音乐生成或第三方托管服务混入。

## 结论

**“主流提供方都是 24 kHz”不成立。**

- 腾讯云“语音技术”的基础、实时 WS 和双向流式 WSv2 默认输出是 **16 kHz**，24 kHz 只是部分音色可选；腾讯云 MPS 新版 WebSocket TTS 则默认 **22.05 kHz** 并按音色协商。腾讯内部也没有统一默认值。
- 火山引擎/豆包的大模型 TTS 多个接口以 **24 kHz** 为默认或官方示例值，但当前流式接口可选范围已扩展到 8/16/22.05/24/32/44.1/48 kHz；不能概括为“只能 24 kHz”。
- MiniMax 同步 HTTP、同步 WebSocket 和异步长文本的默认值都是 **32 kHz**，24 kHz 只是可选值之一。

因此，跨供应商接入必须显式指定并校验采样率。若业务统一为 24 kHz，应确认腾讯云所选音色是否支持 24 kHz，并对腾讯云长文本接口另做 16→24 kHz 重采样或保留 16 kHz。

## 汇总表

| 提供方 / 接口 | 返回方式 | 默认采样率 | 可选采样率 | 格式与默认值 | 重要限制 | 官方来源（查询 2026-07-23） |
|---|---|---:|---|---|---|---|
| 腾讯云 `TextToVoice` 基础合成 | HTTP/HTTPS，整段 Base64 | 16 kHz | 8/16/24 kHz | `wav`（默认）、`mp3`、`pcm` | 24 kHz 仅部分音色 | [基础语音合成 API](https://cloud.tencent.com/document/api/1073/37995) |
| 腾讯云 `TextToStreamAudioWS` | WebSocket；文本一次提交、音频 binary 帧流式返回 | 16 kHz | 8/16/24 kHz | `pcm`、`mp3`；16-bit、mono | 24 kHz 仅部分音色；页面摘要与参数表对 24 kHz 有矛盾，须按参数表及音色表核对 | [实时语音合成 WS](https://cloud.tencent.com/document/api/1073/94308) |
| 腾讯云 `TextToStreamAudioWSv2` | WebSocket；文本增量输入、音频 binary 帧流式返回 | 16 kHz | 8/16/24 kHz | `pcm`（文档注明默认）、`mp3`；16-bit、mono | 24 kHz 仅部分音色；仅明确支持中文/英文音色 | [流式文本语音合成 WSv2](https://cloud.tencent.com/document/api/1073/108595) |
| 腾讯云 `CreateTtsTask` 长文本 | HTTP 异步任务，稍后取得音频 URL | 16 kHz | 8/16 kHz | `mp3`（默认）、`wav`、`pcm` | **不支持 24 kHz** | [长文本语音合成请求](https://cloud.tencent.com/document/api/1073/57373) |
| 腾讯云 MPS WebSocket TTS | WebSocket；文本增量输入、音频 binary 帧流式返回 | 22.05 kHz | 常用 8/16/22.05/24/32/44.1/48 kHz，最终按音色协商 | `mp3`（默认）、`pcm`、`wav`、`flac`、`opus`、`ulaw`、`alaw` | 请求组合不受音色支持时整体回退到该音色默认值，必须以 Handshake 返回为准 | [MPS WebSocket 流式 TTS 协议](https://cloud.tencent.com/document/product/862/133241) |
| 豆包当前单向流式 HTTP | HTTP Chunked，一次输入、流式输出 | 当前参数页未明确写默认采样率；示例为 24 kHz | 8/16/22.05/24/32/44.1/48 kHz | `mp3`（默认）、`pcm`、`ogg_opus`、`wav` | 流式推荐 PCM，不建议 WAV；WAV 会产生流式容器头问题 | [单向流式语音合成 HTTP](https://www.volcengine.com/docs/6561/2528925) |
| 豆包当前单向流式 WebSocket | WebSocket，一次输入、流式输出 | 当前参数页应以请求实际返回校验；官方示例为 24 kHz | 当前公共 `audio_params` 支持 8/16/22.05/24/32/44.1/48 kHz | `mp3`（默认）、`pcm`、`ogg_opus`、`wav` | 流式推荐 PCM，不建议 WAV | [单向流式语音合成 WebSocket](https://www.volcengine.com/docs/6561/2534913) |
| 豆包当前双向流式 WebSocket | WebSocket，文本与音频双流 | 当前参数页应以请求实际返回校验；官方示例为 24 kHz | 8/16/22.05/24/32/44.1/48 kHz | `mp3`（默认）、`pcm`、`ogg_opus`、`wav` | 流式推荐 PCM，不建议 WAV | [双向流式语音合成 WebSocket](https://www.volcengine.com/docs/6561/2532486) |
| 豆包大模型异步长文本 | HTTP submit/query，异步 URL | 24 kHz | 8/16/22.05/24/32/44.1/48 kHz | `mp3`（默认）、`ogg_opus`、`pcm`；传 `wav` 不报错 | 文档警告流式处理 WAV 会重复返回 WAV header；异步结果本身支持 WAV 能力说明，但请求表主枚举为前三者 | [异步长文本接口](https://www.volcengine.com/docs/6561/1829010) |
| 豆包历史 V1 大模型 TTS | HTTP 非流式或 WS 单向流式 | 24 kHz | 8/16/24 kHz | `pcm`（默认）、`mp3`、`ogg_opus`、`wav` | WAV 不支持流式；V1 不支持豆包 TTS 2.0 音色，官方建议大模型使用新接口 | [历史 V1 WebSocket/HTTP 参数](https://www.volcengine.com/docs/6561/2228192) |
| MiniMax `t2a_v2` HTTP 非流式 | HTTP JSON，hex 或 URL | 32 kHz | 8/16/22.05/24/32/44.1 kHz | `mp3`（默认）、`pcm`、`flac`、`wav`、`pcmu_raw`、`pcmu_wav`、`opus` | `pcmu_*` 固定 G.711 μ-law 8 kHz；URL 只用于非流式 | [同步语音合成 HTTP](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http) |
| MiniMax `t2a_v2` HTTP 流式 | HTTP SSE/流式 JSON hex chunks | 32 kHz | 8/16/22.05/24/32/44.1 kHz | 同一 `audio_setting`；流式结果只返回 hex | 官方概览称 WAV 只支持非流式；效果器在流式时仅支持 MP3 | [同步 HTTP API](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http) [接口概览](https://platform.minimaxi.com/docs/api-reference/api-overview) |
| MiniMax `t2a_v2` WebSocket | WebSocket，hex 音频 chunks | 32 kHz | 8/16/22.05/24/32/44.1 kHz | `mp3`（默认）、`pcm`、`flac`、`wav`、`pcmu_raw`、`pcmu_wav`、`opus` | `pcmu_*` 固定 8 kHz；Opus chunk 必须按到达顺序拼接后解码；官方概览的“WAV 仅非流式”与 WS schema 列出 WAV 存在不一致，生产接入应实测或避用 WS+WAV | [同步语音合成 WebSocket](https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket) |
| MiniMax `t2a_async_v2` 长文本 | HTTP 异步任务，下载完整文件 | 32 kHz | 一般格式：8/16/22.05/24/32/44.1 kHz；Opus：8/12/16/24/48 kHz | `mp3`（默认）、`pcm`、`flac`、`wav`、`pcmu_raw`、`pcmu_wav`、`opus` | `pcmu_*` 固定 8 kHz；Opus 使用专属采样率集合；声音效果器仅支持 mp3/wav/flac | [创建异步语音合成任务](https://platform.minimaxi.com/docs/api-reference/speech-t2a-async-create) |

## 腾讯云

### 默认值与接口差异

腾讯云不是 24 kHz 默认体系。基础 HTTP、一次输入/流式输出 WS、文本输入/音频输出双流 WSv2 都把 `16000` 写为默认，`8000` 和 `24000` 为其他选择；长文本异步接口只接受 8/16 kHz。[基础 API](https://cloud.tencent.com/document/api/1073/37995) [实时 WS](https://cloud.tencent.com/document/api/1073/94308) [双流 WSv2](https://cloud.tencent.com/document/api/1073/108595) [长文本](https://cloud.tencent.com/document/api/1073/57373)（均查询于 2026-07-23）

腾讯云 MPS 的另一套 WebSocket TTS 协议不同：`sampleRate` 默认 22.05 kHz，文档列出常用 8/16/22.05/24/32/44.1/48 kHz。服务会按具体音色协商 `(format, sampleRate)`；不支持请求组合时会整体回退，并在 `HandshakeResult` 返回最终格式与采样率。因此接入方不能用请求值代替实际返回值。[腾讯云 MPS WebSocket TTS](https://cloud.tencent.com/document/product/862/133241)（查询于 2026-07-23）

返回形态也不同：

- `TextToVoice` 整段返回 JSON Base64 音频，默认 WAV；请求参数还列出 MP3、PCM。
- `TextToStreamAudioWS` 在握手时一次提交完整文本，之后用 binary frame 连续返回 PCM 或 MP3。
- `TextToStreamAudioWSv2` 可连续发送文本，并持续接收 PCM 或 MP3 binary frame。
- `CreateTtsTask` 是长文本异步任务，完成后经回调或轮询取得音频 URL，默认 MP3。

### 音色限制

24 kHz 必须按音色核对。腾讯云官方音色表按“实时语音合成 / 基础语音合成 / 长文本语音合成”分别列能力，同一 `VoiceType` 不能假定跨接口可用。当前表中精品音色多为 8/16 kHz，大模型和超自然大模型音色通常列 8/16/24 kHz；最终以具体行的“支持采样率”列为准。[腾讯云官方音色列表](https://cloud.tencent.com/document/product/1073/92668)（查询于 2026-07-23）

腾讯云实时 WS 页面顶部仍有“仅 8k/16k”的旧摘要，而同页现行参数表和官方音色表允许部分音色 24 kHz。本文采用更具体的参数表与音色表，但把这一官方页面内部矛盾保留下来，接入时应对目标音色做实际请求验证。[实时 WS 参数页](https://cloud.tencent.com/document/api/1073/94308)（查询于 2026-07-23）

## 火山引擎 / 豆包

### 当前大模型同步流式接口

2026-07-20 更新的当前单向流式 HTTP 参数页给出的采样率集合是 `8000, 16000, 22050, 24000, 32000, 44100, 48000`，格式是 MP3/PCM/Ogg Opus/WAV，默认格式 MP3；官方示例使用 24 kHz，但该页没有把 `sample_rate` 的默认值写进参数定义，所以不应只凭示例把默认值硬断言为 24 kHz。[当前单向流式 HTTP](https://www.volcengine.com/docs/6561/2528925)（查询于 2026-07-23）

当前单向和双向 WebSocket 文档沿用同一 `audio_params` 范围与 24 kHz 示例。三种流式协议都建议使用 PCM，不建议 WAV；WAV 是有容器头的格式，流式拼接容易出现重复 header。[单向 WS](https://www.volcengine.com/docs/6561/2534913) [双向 WS](https://www.volcengine.com/docs/6561/2532486)（查询于 2026-07-23）

### 异步、历史接口与模型限制

大模型异步长文本的参数表明确默认 24 kHz，可选 8/16/22.05/24/32/44.1/48 kHz，默认 MP3；支持公版、复刻和混合音色。[大模型异步长文本](https://www.volcengine.com/docs/6561/1829010)（查询于 2026-07-23）

历史 V1 大模型接口则是默认 24 kHz、另可选 8/16 kHz，格式为 PCM（默认）/MP3/Ogg Opus/WAV，且 WAV 不支持流式。官方同时注明 V1 不支持豆包语音合成模型 2.0 的音色，大模型音色推荐迁移新接口。[历史 V1 接口](https://www.volcengine.com/docs/6561/2228192)（查询于 2026-07-23）

旧版产品能力页曾写单向/非流式仅 8/16/24 kHz、双向流式增加 48 kHz；2026-07 的新接口参数页已列出更大的集合，应以实际采用接口的最新参数页为准，不能拿旧页限制覆盖新接口。[豆包大模型产品能力页](https://www.volcengine.com/docs/6561/1257543)（查询于 2026-07-23）

小模型异步长文本接口也将 24 kHz 写为默认，格式支持 PCM/WAV/MP3/Ogg Opus，但语种、情感和音色需要按小模型音色列表核对。[小模型异步长文本](https://www.volcengine.com/docs/6561/1096680)（查询于 2026-07-23）

## MiniMax

### 同步 HTTP 与 WebSocket

MiniMax `t2a_v2` 的同步 HTTP 和 WebSocket 都将采样率默认值写为 **32,000 Hz**，可选 `8000, 16000, 22050, 24000, 32000, 44100`；默认格式为 MP3。[HTTP API](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http) [WebSocket API](https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket)（查询于 2026-07-23）

同步 HTTP 可由 `stream=false/true` 控制整段或流式输出。非流式 `output_format` 可选 hex 或 URL，默认 hex；流式只支持 hex chunks。WebSocket 同样返回 hex 音频块。[HTTP 输出规则](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http) [WebSocket 消息](https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket)（查询于 2026-07-23）

通用格式 schema 列出 MP3、PCM、FLAC、WAV、`pcmu_raw`、`pcmu_wav`、Opus。其中 `pcmu_*` 是 G.711 μ-law，固定 8 kHz；所以即使同时传了其他 `sample_rate`，也不能认为 G.711 会输出该采样率。WebSocket 文档还要求 Opus chunks 按到达顺序拼接再解码。[MiniMax WebSocket `audio_setting`](https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket)（查询于 2026-07-23）

官方接口概览称 WAV 只支持非流式，但 WebSocket 的当前 schema 又把 WAV 放在枚举中；这是官方文档内部不一致。稳妥做法是流式使用 MP3、PCM、FLAC 或 Opus，并在确需 WAV 时先做实际 API 验证。[MiniMax 接口概览](https://platform.minimaxi.com/docs/api-reference/api-overview) [WebSocket schema](https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket)（查询于 2026-07-23）

### 异步长文本

`t2a_async_v2` 默认 32 kHz、默认 MP3。一般格式支持 8/16/22.05/24/32/44.1 kHz；Opus 是特殊集合 8/12/16/24/48 kHz，其他值会报错。`pcmu_raw`/`pcmu_wav` 固定 8 kHz。若启用 `voice_modify`，输出格式限制为 MP3/WAV/FLAC。[MiniMax 异步 T2A API](https://platform.minimaxi.com/docs/api-reference/speech-t2a-async-create)（查询于 2026-07-23）

当前同步和异步 API 的模型枚举覆盖 `speech-2.8-*`、`speech-2.6-*`、`speech-02-*`、`speech-01-*`。采样率与基本格式字段没有按系统音色、复刻音色或具体模型列出不同集合；官方明确列出的差异主要是模型支持的语气词、语言、字幕或声音效果器，而不是采样率。因此不能自行推导“某个音色天然 24 kHz”，应以返回的 `extra_info.audio_sample_rate` 为最终校验。[MiniMax 同步 T2A 模型与响应字段](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http)（查询于 2026-07-23）

## 接入建议

1. 请求中显式传采样率和格式，不依赖供应商默认值。
2. 收到首个完整响应或首个可解码流后读取实际采样率；MiniMax 可直接校验 `extra_info.audio_sample_rate`，其他接口可解析容器头或结合请求元数据校验。
3. 对腾讯云建立“接口 × VoiceType × 采样率”白名单；24 kHz 不是所有音色通用能力。
4. 对流式接口优先使用裸 PCM 或供应商明确推荐的编码。WAV 容器不适合简单拼接多个流式块。
5. 电话链路通常使用 8/16 kHz；如果最终仍要降采样，供应商输出 24/32/48 kHz 并不自动带来端到端收益。
