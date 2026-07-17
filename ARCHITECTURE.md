# ARCHITECTURE

会悟采用本地优先的前端单页应用 + FastAPI 后端 + 本地模型目录结构。

## 目标分层

```text
frontend/
  UI 展示层
  状态机与交互层
  API / WebSocket 客户端层

backend/
  api/                HTTP / WebSocket 路由
  services/           模型运行时、状态、上传任务、记录、摘要
  xasr/               X-ASR、Qwen3-ASR、VAD、实时音频协议
  diarization/        说话人分离
  modules/            文本增强、LLM、持久化适配

models/
  xasr/
  vad/
  qwen3/
  diarization/
```

## 当前接口契约

外部接口保持兼容：

- `GET /api/health`
- `GET /api/xasr/status`
- `GET /api/settings`
- `PUT /api/settings`
- `POST /api/audio/upload`
- `WS /ws/upload/{file_id}`
- `WS /ws/live`

内部重构不得改变这些接口的路径和基本响应语义。

## 模型路径

所有模型路径由 `backend/xasr/model_paths.py` 统一解析：

- `models/xasr/`
- `models/vad/silero_vad.onnx`
- `models/qwen3/`
- `models/diarization/`

显式 `DITING_*` 环境变量优先，`HUIWU_MODELS_DIR` 作为统一根目录，默认指向项目根目录 `models/`。

## ASR 流程

### 实时麦克风

```text
Browser AudioWorklet
  -> 16 kHz mono PCM
  -> DTP2 WebSocket binary frame
  -> /ws/live
  -> LiveAudioSession
  -> live VAD
  -> X-ASR partial/final preview
  -> complete WAV recording
  -> optional canonical final transcript
```

实时 partial 始终使用 X-ASR。Qwen3-ASR 不进入实时 partial 链路。

### 上传文件

```text
/api/audio/upload
  -> save_upload_to_temp
  -> UploadJob
  -> file VAD
  -> X-ASR or Qwen3 final engine
  -> optional diarization
  -> record store
  -> HTTP response + WS progress events
```

上传任务应从 WebSocket 生命周期中解耦：WebSocket 只订阅进度，不拥有任务本身。

## 模型状态

`/api/xasr/status` 是前端、doctor 和人工排查的统一状态入口，应包含：

- `paths`
- `profiles`
- `providers`
- `live_vad`
- `file_vad`
- `diarization`
- `resources`
- `features`

缺模型或可选依赖时应显示降级原因，不应让接口失败。

## 前端结构

当前 [frontend/index.html](frontend/index.html) 仍承担大量职责。目标是保持原生前端并逐步拆分：

```text
frontend/css/
  base.css
  layout.css
  transcription.css
  settings.css
  mic-orb.css

frontend/js/
  api-client.js
  live-mic-client.js
  live-mic-controller.js
  mic-orb.js
  upload-controller.js
  settings-controller.js
  records-controller.js
  app-init.js
```

现有可复用模块：

- `frontend/app-settings.js`
- `frontend/live-protocol.js`
- `frontend/audio-worklet.js`
- `frontend/mic-level.js`
- `frontend/management-transcription.js`

## 麦克风状态机

目标状态：

```text
idle
connecting
recording
stopping
finalizing
complete
error
```

所有按钮、语音球、状态栏、WebSocket 回调都应由状态机驱动。旧 WebSocket message 必须通过 session token guard 过滤，避免停止旧会话时清理新会话。

## 高风险目录策略

本轮不删除、不移动以下目录，只通过 `.gitignore`、doctor 和文档说明管理：

- `_ref_dickrun/`：参考仓库副本，不参与运行。
- `.idea/`：IDE 本地配置。
- `backend/xasr/zipformer/`：训练/导出/上游参考源码，不属于产品运行时主链路。
- `backend/recordings/`：用户录音数据。
- `models/`：用户本地模型权重。
- `backend/data/`：数据库、设置、密钥等运行时状态。
- `backend/logs/`：运行日志。

## 验收标准

- `python tools/doctor.py` 能明确指出缺失模型和依赖。
- `python start.py` 打印路径与 `/api/xasr/status.paths` 一致。
- 设置页能看到 X-ASR、VAD、Qwen3、diarization 真实状态。
- 麦克风支持连续开始/停止/开始/停止。
- 上传短音频和长音频能显示 provider、VAD、RTF 和 fallback reason。
- 缺 diarization 模型时 UI 显示 ASR-only。
- README、TECHNICAL_DOCS、MODEL_SETUP 路径一致。
