# 会悟 v3.0 — 技术文档

本文描述当前分支实际实现的系统边界、语音链路、模型切换、数据持久化和接口契约。产品正式名称为“会悟”。`DITING_*`、`diting.db`、`diting.log`、`DiTing*` 等名称仅作为历史兼容接口保留，重命名会破坏已有部署或数据，因此不属于品牌残留清理范围。

## 1. 系统边界

会悟由浏览器单页前端和 FastAPI 后端组成，核心能力分为四层：

1. **会议协作层**：账号、用户资料、同事关系、共享预约和会议分析。
2. **语音处理层**：浏览器采集、上传、统一重采样、VAD、X-ASR/Qwen3-ASR 和说话人分离。
3. **记录认知层**：分段音频、文本、质量元数据、热词、检索、摘要和文字流程图。
4. **运行保障层**：健康检查、运行时模型状态、日志、兼容配置和失败回退。

```text
frontend/index.html + audio-worklet.js
        │ HTTP / WebSocket / DTP2 PCM
        ▼
backend/main.py
        ├─ xasr/ + diarization/          语音识别与说话人日志
        ├─ modules/                      文本增强、LLM、摘要
        ├─ modules/*_store.py            用户、预约、记录、摘要、设置
        └─ utils/logger.py               控制台、文件、环形日志
```

后端 API revision 为 4。启动器会检查 revision，阻止旧服务占用端口时被误认为当前实例。

## 2. 核心目录

| 路径 | 职责 |
|---|---|
| `frontend/index.html` | 转写、预约、记录、用户和设置界面 |
| `frontend/audio-worklet.js` | 麦克风采集、连续降采样、声量统计 |
| `backend/main.py` | FastAPI 路由、WebSocket 会话和任务编排 |
| `backend/audio_buffer.py` | 16 kHz 单声道 canonical audio 时间轴 |
| `backend/xasr/asr_engine.py` | X-ASR 文件/流式识别与结果构建 |
| `backend/xasr/qwen3_engine.py` | Qwen3-ASR 本地模型加载和最终转写 |
| `backend/xasr/engine_pool.py` | 运行时预热、模型选择、状态和回退 |
| `backend/xasr/live_audio.py` | PCM 校验、Silero VAD、端点和录音发布 |
| `backend/diarization/pipeline.py` | 说话人双轨分析、跨块统一、边界重识别 |
| `backend/modules/audio_processor.py` | 质量、热词、逻辑和不确定性 |
| `backend/modules/summary_service.py` | 动态摘要与 Mermaid/Markdown 文字图 |
| `backend/modules/meeting_db.py`、`record_store.py`、`summary_store.py` | SQLite/Supabase 和用户级设置持久化 |

## 3. 语音识别工作流

### 3.1 统一输入

无论浏览器麦克风还是上传文件，后端分析模块最终都使用 16 kHz、单声道、float32/PCM16 的统一时间轴。ASR、VAD、diarization、波形片段和质量计算共享同一坐标系，避免模块各自解码、重采样引起的边界漂移。

### 3.2 实时麦克风链路

```text
MediaDevices.getUserMedia
  → AudioWorklet（设备真实采样率）
  → 连续降采样 16 kHz mono
  → DTP2 + uint32 sequence + pcm_s16le
  → /ws/live
  → 帧序号/幅值校验
  → Silero VAD（缺失时能量 VAD）
  → X-ASR OnlineRecognizer
  → live_result / final_result
```

实时链路固定使用 X-ASR，以避免生成式大模型的加载和推理延迟阻塞交互。Silero 检测到候选句尾后进入可撤销宽限期：宽限期内恢复说话则继续同一句，持续静音才真正定稿。断线录音先保留为 `.wav.part`，正常结束后原子发布 `.wav`。

### 3.3 最终转写链路

```text
完整 WAV
  → 统一解码/重采样
  → 所选 provider：X-ASR 或 Qwen3-ASR
  → 失败时记录 provider_error 并回退 X-ASR
  → 说话人时间线对齐
  → 热词、标点、ITN、逻辑与不确定性
  → canonical final_transcript
  → 覆盖实时预览并可保存为会议记录
```

Qwen3-ASR 只负责上传或停止录音后的最终识别。设置保存后，前端等待运行时真正加载完成；状态中的 `selected_provider` 表示用户选择，`effective_provider` 表示实际执行引擎，`provider_fallback` 表示是否发生降级。

### 3.4 Qwen3-ASR 运行条件

- 安装 `qwen-asr` 和与显卡匹配的 CUDA PyTorch。
- 模型目录包含配置、tokenizer/preprocessor 和完整权重分片。
- 设备可选 CUDA 0，常用精度为 `bfloat16`；硬件不支持时应改用可用精度或 X-ASR。
- 模型加载失败、推理异常或输出为空时，完整 WAV 仍被保留并由 X-ASR 重试。

会悟不宣称 Qwen3-ASR 是低延迟流式模型；当前可切换范围是“最终稿引擎”，不是会中 partial 引擎。

## 4. VAD、说话人和文本增强

### 4.1 文件 VAD 与连续识别

VAD 段只表示语音活动时间，不直接等同于说话人或独立 ASR 会话。X-ASR 在一条连续流中保留声学上下文，再用累计文本和完整句标点映射回时间范围，降低短句、轻声和语气词因频繁重置而丢失的概率。

### 4.2 说话人双轨分析

```text
AudioBuffer ──→ 连续 X-ASR ─────────────────────┐
      └──────→ pyannote segmentation             │
                → 3D-Speaker embedding           │
                → 聚类/平滑/跨块统一             │
                              ↓                   │
                      speaker timeline ──────────┤
                                                  ↓
                            时间重叠对齐与换人边界局部重识别
```

已知人数可用 2～8 人约束聚类。超过阈值的长音频采用静音感知分块，各块先生成局部说话人，再通过声纹向量进行全局聚类。重叠语音当前只标记 `overlap` 并降低置信度，不伪装为已完成双路分离。

### 4.3 文本增强

结果可包含：

- 热词解码增强、逐词权重和模糊拼音纠错；
- 标点恢复、重复清理、中文数字 ITN；
- SNR/RT60 和质量标签；
- 数字、百分比等逻辑冲突标记；
- 低置信度或可疑文本范围。

热词保存在 `backend/data/hotwords.json`，运行设置保存在 `backend/data/settings.json`，新会话读取最新配置。

## 5. 大模型与摘要

`llm_client.py` 使用 OpenAI Chat Completions 兼容协议。仓库默认 Provider 为 DeepSeek，默认模型 ID 为 `deepseek-v4-flash`，并预置 `deepseek-v4-pro`、Qwen3.7/Qwen3.6 和 GPT-5.x 等文字模型。自定义网关可填写任意模型 ID，例如 `dsv4pro`。

模型列表和连接验证是两种不同操作：

- `POST /api/llm-settings/models`：读取网关 `/models`。
- `POST /api/llm-settings/test`：执行一次真实、受约束的 JSON 推理。

API Key 使用 Fernet 加密，密钥位于未提交的 `backend/data/llm_secret.key`。摘要记录只保留脱敏配置快照。

### 5.1 动态章节

摘要服务要求模型返回结构化字段，但不要求最终 Markdown 固定展示所有章节。某字段为空时，对应章节不渲染；剩余章节按实际内容连续编号。文字结构图只有在存在有效节点/边时追加。

### 5.2 不支持图像的模型

DSv4 等文字模型不会被路由到图像生成接口。模型只产生文字节点与关系，本地服务将其转换为 Mermaid 和 Markdown。这样既兼容纯聊天模型，也使会议结构图可审计、可编辑、可版本化。

## 6. 数据与权限

### 6.1 管理数据库

默认 `backend/data/diting.db`（历史兼容文件名）保存用户、会话、同事、预约、参会关系和会议分析。SQLite 启用外键、索引、busy timeout 和 WAL；配置 Supabase 后可切换云端存储。

### 6.2 记录数据库

默认 `backend/data/records.db` 保存：

- `meeting_records`：标题、来源、原文件、说话人、完整文本、时长、所有者和状态。
- `meeting_record_segments`：时间、说话人、文本、识别元数据和分段 WAV BLOB。
- 摘要及摘要条目：状态、Markdown、结构化字段和脱敏模型快照。

列表查询不加载音频 BLOB；读取详情或源音频时才加载大字段。普通用户只能访问自己的数据，管理员按接口权限查看全局数据。

### 6.3 兼容标识

| 标识 | 保留原因 |
|---|---|
| `DITING_*` | 已有环境变量和部署脚本依赖 |
| `backend/data/diting.db` | 直接改名会导致旧数据不可见 |
| `backend/logs/diting.log` | 运维脚本和历史日志路径依赖 |
| `DiTing*` / `diting-*` 前端键 | 浏览器存储、Worklet 和运行时协议依赖 |

新文档和界面只使用“会悟”品牌；以上标识只在技术语境中出现。

## 7. HTTP API

以下为主要路由族，完整参数与响应以运行时 `/docs` 为准。

| 路由族 | 主要用途 |
|---|---|
| `GET /api/health` | API revision、模型与运行状态 |
| `/api/auth/*` | 登录、注销、当前用户资料 |
| `/api/users/*`、`/api/friends/*` | 用户管理和同事关系 |
| `/api/meetings/reservations*` | 共享预约 CRUD |
| `/api/meetings/analysis*` | 转写分析持久化与查询 |
| `POST /api/audio/upload` | 文件上传与最终处理 |
| `GET/PUT /api/settings` | 识别、VAD、麦克风、热词统一设置 |
| `GET/POST/PUT /api/hotwords` | 热词兼容读取、追加和完整替换 |
| `/api/records*` | 会议记录、分段、源音频、定稿 |
| `/api/record-summaries*` | 摘要创建、查询、重试与下载 |
| `/api/llm-settings*` | 用户模型设置、模型列表和真实连接测试 |
| `/api/xasr/status` | 识别 provider、模型和回退状态 |
| `/api/logs/*`、`/api/eval/*` | 日志与 Eval_Ali 评测辅助 |

记录流程支持先创建草稿，再增量写入分段，最后调用 finalize 生成完整文本和统计；避免长录音结束时一次性提交全部数据。

## 8. WebSocket 协议

| 端点 | 方向 | 用途 |
|---|---|---|
| `/ws/live` | 双向 | 麦克风 PCM、实时/最终结果、停止和错误 |
| `/ws/upload/{file_id}` | 后端到前端 | 上传阶段、分段、进度和完成状态 |
| `/ws/logs` | 后端到前端 | 调试日志流 |

实时二进制帧含 DTP2 魔数和递增序号。服务端据此检测丢帧、乱序和异常幅值。停止消息后，服务端先补齐流式句尾，再发布完整 WAV 和执行最终转写；前端必须以最终结果替换 partial，而不是追加重复文本。

## 9. 配置

常用环境变量：

```powershell
$env:DITING_MAX_UPLOAD_MB = "2048"
$env:DITING_PROCESSING_WORKERS = "2"
$env:DITING_LIVE_ENDPOINT_GRACE_MS = "800"
$env:DITING_DIARIZATION_CHUNKING = "true"
$env:DITING_DIARIZATION_MAX_WORKERS = "2"
$env:DITING_LLM_PROVIDER = "deepseek"
$env:DITING_LLM_MODEL = "deepseek-v4-flash"
```

模型延迟档位包括 `low-latency`（160 ms）、`balanced`（480 ms）、`meeting`（960 ms）和 `quality`（1920 ms）；只有对应 ONNX 文件存在时才可选择。实时和最终 X-ASR 档位相同时共享已预热运行时。

## 10. 部署与验证

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
python backend\xasr\download_models.py --profile meeting
python start.py
```

测试：

```powershell
.venv\Scripts\python.exe -m pytest backend\tests -q
node --test frontend\tests\*.test.mjs
```

协议烟测：

```powershell
python backend\tests\smoke_live_websocket.py sample.wav --url ws://127.0.0.1:8765/ws/live
```

运行时检查顺序：

1. `GET /api/health` 确认 revision 和后端实例。
2. `GET /api/xasr/status` 核对 selected/effective provider 和 fallback 原因。
3. 设置页执行模型加载或 LLM 真实连接测试。
4. 日志面板或 `backend/logs/diting.log` 查看阶段错误。
5. 麦克风问题先检查浏览器权限、设备、RMS，再检查 WebSocket 帧和 VAD。

## 11. 安全与隐私

- 本地模式可在无外部语音 API 的环境运行，音频和模型留在本机。
- 使用外部大模型摘要会发送转写文本；部署者应明确告知用户并选择合规网关。
- Supabase 示例 RLS 只用于开发，公网部署必须使用后端服务端密钥和最小权限策略。
- 不提交 `.env`、API Key、模型权重、录音、数据库和日志。
- 头像 data URL 有类型和解码后大小限制；普通用户不可修改角色、状态和账号 ID。

## 12. 已知限制与路线图

| 当前限制 | 影响 | 规划 |
|---|---|---|
| Qwen3-ASR 不参与实时 partial | 会中仍由 X-ASR 输出 | 保持双阶段边界，优先优化会后时延 |
| 重叠语音只标记不分离 | 同时发言文字仍可能混合 | 对重叠区按需分离并回绑时间轴 |
| 匿名说话人标签 | 无法自动关联参会人 | 增加授权声纹注册与匹配 |
| 前端单文件较大 | 模块耦合与测试成本高 | 拆分录音、设置、记录、预约模块 |
| 本地模型资源占用高 | 低显存设备可能回退 | 增加量化档位和显存/时延基准 |
| 公网部署策略仍需收紧 | 不适合直接暴露开发配置 | 增加审计、密钥轮换与对象存储 |

## 附录：日志与仓库

```powershell
Get-Content backend\logs\diting.log -Wait
```

- 仓库：<https://github.com/ding060426/dickrun>
- 模型、音频、数据集、数据库和日志不进入 Git 提交。
- 文档版本：会悟 v3.0，更新日期：2026-07-16。
