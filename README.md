# 会悟 v3.0 — 智能会议语音认知系统

会悟是一套面向真实会议场景的本地优先语音认知产品。它把会议预约、多人录音、实时转写、会后精转、说话人日志、记录管理和大模型摘要连接为一条可追溯工作流，帮助团队把“开过的会”转化为可检索、可复盘、可执行的知识资产。

> 正式产品名统一为“会悟”。仓库仅保留 `DITING_*` 环境变量、`diting.db` 和 `diting.log` 等外部兼容契约，以保证已有部署和数据继续可用。前端运行时对象、AudioWorklet 和内部线程/临时目录均已改用 `HuiWu` / `huiwu`；旧浏览器登录与语言键只在首次读取时迁移并立即删除。

## 产品定位

会悟面向课程研讨、项目例会、需求评审、访谈调研、远程协作和政企内网会议等场景，重点解决四类问题：

- 会中信息易遗漏：提供低延迟麦克风预览、文件转写和真实音频波形反馈。
- 多人内容难归属：通过说话人分离、换人边界重识别和标签重命名形成结构化发言日志。
- 会后整理成本高：保存原始录音、分段文本与认知元数据，并按内容动态生成摘要。
- 数据与模型不可控：核心语音链路可完全本地运行，X-ASR 与 Qwen3-ASR 可按设备能力切换。

## 核心工作流

```text
登录 / 预约会议
        ↓
麦克风实时录音或上传文件
        ↓
16 kHz 单声道统一时间轴
        ├─ 实时链路：AudioWorklet → DTP2 WebSocket → Silero VAD → X-ASR 预览
        └─ 最终链路：完整 WAV → X-ASR 或 Qwen3-ASR → 说话人分离 → 文本增强
        ↓
保存会议记录、分段音频、说话人和质量信息
        ↓
DSv4 / Qwen / OpenAI 兼容模型生成动态摘要与 Mermaid 文字流程图
```

系统刻意区分“实时预览”和“最终稿”：实时链路优先保证反馈速度；停止录音后再使用完整 WAV 和设置中选择的最终识别引擎生成 canonical transcript。若 Qwen3-ASR 加载或推理失败，后端会记录原因并回退 X-ASR，不会丢失录音。

## 主要创新

- **双阶段识别**：低延迟 X-ASR 负责会中反馈，X-ASR/Qwen3-ASR 负责会后最终稿，兼顾交互速度和模型能力。
- **统一音频时间轴**：ASR、VAD、说话人分离和波形均复用同一份 16 kHz 单声道音频，减少跨模块时间戳漂移。
- **边界感知的多人转写**：说话人分析与连续 ASR 双轨执行，只在换人边界对必要片段局部重识别，避免逐段重置造成吞字。
- **动态会议摘要**：无内容的章节自动省略，后续序号连续重排；文字模型只生成结构化节点，由本地代码渲染 Mermaid/Markdown，不调用图像接口。
- **本地优先与可降级**：模型、记录、热词和设置均可本地保存；可选能力不可用时提供明确状态和安全回退。

## 项目结构

```text
frontend/
  index.html                  单页产品界面：转写、预约、记录、用户与设置
  audio-worklet.js            浏览器采集与连续降采样
backend/
  main.py                     FastAPI HTTP / WebSocket 入口
  audio_buffer.py             统一音频时间轴
  xasr/
    asr_engine.py             X-ASR 流式识别
    qwen3_engine.py           Qwen3-ASR 最终转写适配器
    engine_pool.py            最终识别模型选择与回退
    live_audio.py             麦克风 PCM、VAD 与端点状态机
  diarization/                pyannote segmentation + 3D-Speaker
  modules/
    audio_processor.py        SNR、热词、逻辑校验与不确定性
    llm_client.py             OpenAI Chat Completions 兼容客户端
    llm_models.py             DSv4/Qwen/OpenAI 模型目录与能力路由
    summary_service.py        动态摘要与文字流程图
    meeting_db.py / record_store.py / summary_store.py
                              用户、预约、记录、摘要和设置持久化
docs/
  会悟智能会议语音认知系统实验报告.docx
```

更完整的接口、存储、协议和故障排查见 [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)，会议记录生命周期见 [MEETING_RECORDS.md](MEETING_RECORDS.md)。

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 2. 下载 X-ASR 模型

```powershell
python backend\xasr\download_models.py --profile meeting
```

默认模型目录：

```text
backend/xasr/models/
  encoder-960ms.onnx
  decoder-960ms.onnx
  joiner-960ms.onnx
  tokens.txt
  silero_vad.onnx
```

可选说话人模型：

```text
backend/diarization/models/
  pyannote-segmentation-3.0.int8.onnx
  3dspeaker-eres2net.onnx
```

### 3. 启动

```powershell
python start.py
```

- 前端：<http://localhost:3000>
- 后端：<http://localhost:8765>
- API 文档：<http://localhost:8765/docs>

启动器会校验后端 API revision。若端口被旧进程占用，会明确提示，而不会把不兼容服务误判为启动成功。

## 本地 Qwen3-ASR

设置页可在 `X-ASR` 与 `Qwen3-ASR` 之间切换最终转写引擎。实时麦克风预览始终使用低延迟 X-ASR；上传文件和停止录音后的最终转写才调用 Qwen3-ASR。

本地识别默认使用 12 个推理线程，可通过环境变量调整：

```powershell
$env:DITING_ASR_THREADS = "12"
python start.py
```

该值会传给 X-ASR recognizer，并设置 Qwen 音频预处理所用的 PyTorch CPU 线程数；Qwen 的主要生成阶段仍受 GPU 显存和算力限制。`DITING_PROCESSING_WORKERS` 是并发会议任务数，不应为了增加单条录音速度而同步调高。切换出 Qwen、CUDA OOM 或后端关闭时，系统会释放 Qwen 模型引用并清理 PyTorch CUDA allocator 缓存。

建议使用独立 Python 3.12 环境：

```powershell
py -3.12 -m venv .venv-qwen3
.venv-qwen3\Scripts\python.exe -m pip install -r backend\requirements.txt qwen-asr
# 再按 PyTorch 官方说明安装与本机 CUDA 匹配的 torch
```

在 `会议转写 → 设置 → 识别与分段` 中填写本地模型目录。目录应包含 `config.json`、tokenizer/preprocessor 文件和完整 `.safetensors` 分片。保存后查看：

```text
selected_provider=qwen3
effective_provider=qwen3
provider_fallback=false
```

若依赖、模型或显存不可用，状态会显示具体原因并回退 X-ASR。Qwen3-ASR 的模型布局以 [官方仓库](https://github.com/QwenLM/Qwen3-ASR) 为准，CUDA 安装以 [PyTorch 官方页面](https://pytorch.org/get-started/locally/) 为准。

## 大模型摘要

在 `记录管理 → 大模型设置` 中配置 Provider、Base URL、模型 ID 和 API Key。当前仓库预置：

- DeepSeek：`deepseek-v4-flash`（默认，界面名 DSv4 Flash）、`deepseek-v4-pro`（DSv4 Pro）。
- Qwen：Qwen3.7 Max/Plus、Qwen3.6 Flash。
- OpenAI 兼容：GPT-5.2、GPT-5.1、GPT-5 mini，或自定义网关模型 ID。

“更新模型列表”读取当前 Base URL 的 `/models`；“测试连接”执行一次真实 JSON 推理。用户设置会用于该用户的摘要任务，API Key 以 Fernet 加密，密钥文件位于忽略提交的 `backend/data/llm_secret.key`。

```powershell
$env:DITING_LLM_PROVIDER = "deepseek"
$env:DITING_LLM_BASE_URL = "https://api.deepseek.com"
$env:DITING_LLM_API_KEY = "<your-key>"
$env:DITING_LLM_MODEL = "deepseek-v4-flash"
python start.py
```

DSv4 等不支持图像生成的聊天模型只处理文字与结构化节点。系统在本地渲染 Mermaid 和 Markdown，不向图像生成接口发送请求。摘要不强制固定章节：议题、决策、行动项、风险等内容为空时直接省略，剩余章节自动连续编号。

## 会议管理与记录

登录后可使用会议转写、会议预约、记录管理、会议分析和用户管理。共享预约仅对创建者、参与者和管理员可见；创建者与管理员可以编辑参会人。

管理数据默认位于 `backend/data/diting.db`，记录数据默认位于 `backend/data/records.db`。前者文件名是历史兼容标识。部署 Supabase 时先执行 `backend/supabase_init.sql`，并把密钥仅保留在后端；示例 RLS 策略只适合开发环境。

## 实时麦克风与离线转写

浏览器 AudioWorklet 按设备真实采样率采集，连续降采样至 16 kHz 单声道 PCM，通过带 DTP2 和序号的 WebSocket 帧发送到 `/ws/live`。停止时发布完整 WAV，再执行最终识别和离线说话人对齐。

无物理麦克风时可用文件验证同一协议：

```powershell
python backend\tests\smoke_live_websocket.py path\to\meeting.wav --url ws://127.0.0.1:8765/ws/live
```

设置、热词和录音分别保存在：

```text
backend/data/settings.json
backend/data/hotwords.json
backend/recordings/
```

## 功能状态

| 功能 | 状态 | 说明 |
|---|---|---|
| 实时 X-ASR | 已实现 | AudioWorklet + DTP2 + Silero VAD + partial/final |
| Qwen3-ASR 最终转写 | 已实现 | 本地 CUDA 模型，可切换并自动回退 |
| 说话人日志 | 已实现 | 全局聚类、时间对齐、边界重识别、标签重命名 |
| 热词与文本增强 | 已实现 | 逐词权重、模糊拼音、标点、ITN、逻辑检查 |
| 本地会议记录 | 已实现 | 分段音频、文本、元数据、检索和导出 |
| DSv4 动态摘要 | 已实现 | 用户级配置、真实连接测试、动态章节 |
| Mermaid 文字流程图 | 已实现 | 结构化文字节点，本地渲染 |
| 重叠语音双路分离 | 规划中 | 当前只标记重叠并降低归属置信度 |
| 注册声纹身份识别 | 规划中 | 当前输出匿名说话人标签 |

## 验证

```powershell
.venv\Scripts\python.exe -m pytest backend\tests -q
node --test frontend\tests\*.test.mjs
```

本次品牌与文档更新提交前会重新执行全量测试；可复现结果以当前分支提交记录和终端输出为准。

## 未来规划

1. 对重叠区段进行按需语音分离，并把分离结果回绑原时间轴。
2. 增加可授权的声纹注册库，实现匿名标签到已知参会人的映射。
3. 将单文件前端拆分为可测试模块，降低设置、录音和记录流程的耦合。
4. 增加长会基准、GPU 显存基线和多模型质量/时延对比报告。
5. 完善公网部署的权限、审计、密钥轮换和对象存储策略。

## License

MIT
