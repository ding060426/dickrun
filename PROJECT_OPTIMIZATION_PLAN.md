# 会悟项目优化方案

## 1. 当前状态判断

会悟当前已经形成一条完整的本地优先会议语音链路：前端单页应用负责会议转写、上传、预约、记录、设置和语音球交互；后端 FastAPI 负责 HTTP / WebSocket 接口、X-ASR 实时识别、上传文件转写、Qwen3-ASR 最终稿、说话人分离、热词、记录持久化和摘要生成。

当前项目的主要优势是：

- 前端和后端接口基本保持 `main` 风格，业务入口集中，便于演示和集成。
- X-ASR、Qwen3-ASR、Silero VAD 已按 `models/` 统一目录迁移，避免模型散落在 `download/` 或后端源码目录。
- 实时麦克风链路已经使用 `AudioWorklet + DTP2 + WebSocket`，比传统 `ScriptProcessorNode` 更稳定。
- 后端已有 `RuntimeConfigStore`、`AsrEnginePool`、`LiveAudioSession` 等关键抽象，具备继续工程化的基础。
- 前端已把上传、实时录音、历史记录、摘要和设置整合到同一页面，适合比赛或产品原型演示。

主要问题是：

- 文档、启动脚本、模型目录和实际代码曾多次迁移，仍存在路径认知不一致风险。
- `frontend/index.html` 过大，HTML、CSS、业务 JS、状态机混在单文件中，后续修改容易引入状态串线。
- `backend/main.py` 承担过多职责，路由、模型编排、记录、摘要、用户、WebSocket 都集中在一个文件。
- 识别链路可运行，但调试观测仍偏日志式，缺少标准化诊断页面和端到端自检命令。
- Qwen3-ASR、说话人分离模型、LLM 摘要属于可选重资源能力，需要更明确的模型选择策略和降级策略。

---

## 2. 推荐目标架构

### 2.1 分层目标

建议把项目稳定为四层：

```text
frontend/
  UI 展示层
  状态机与交互层
  API / WebSocket 客户端层

backend/
  api/                HTTP / WebSocket 路由
  services/           业务编排：转写、记录、摘要、用户、设置
  xasr/               ASR、VAD、Qwen3、实时音频协议
  diarization/        说话人分离
  modules/            文本增强、LLM、存储适配

models/
  xasr/               X-ASR ONNX 模型
  vad/                Silero / FireRed VAD
  qwen3/              Qwen3-ASR 本地权重
  diarization/        pyannote + 3D-Speaker，可选迁移目标

data/runtime/
  settings、hotwords、records、logs、recordings，可选后续统一
```

短期不必一次性重构到位，但新增代码应朝这个方向组织。

### 2.2 保持接口稳定

继续遵守以下原则：

- 外部 API 路由继续保持 `main` 风格，不随内部重构变化。
- `DITING_*` 环境变量继续保留，避免破坏已有配置。
- 前端仍通过 `/api/settings`、`/api/xasr/status`、`/ws/live`、`/api/audio/upload`、`/ws/upload/{file_id}` 等现有契约工作。
- 内部目录变化必须通过路径解析层适配，不能让用户手动猜路径。

---

## 3. 模型选择优化方案

### 3.1 X-ASR 模型策略

当前本地实际部署的是 `160ms` profile：

```text
models/xasr/
  encoder-160ms.onnx
  decoder-160ms.onnx
  joiner-160ms.onnx
  tokens.txt
```

因此当前默认 `low-latency` 是合理的。建议模型策略如下：

| 场景 | 推荐模型 | 原因 |
|---|---|---|
| 实时麦克风预览 | X-ASR `low-latency` / 160ms | 启动快、延迟低、适合 partial 反馈 |
| 短音频上传 | X-ASR `low-latency` 或 Qwen3-ASR | 若追求速度用 X-ASR，若追求最终文本质量用 Qwen3 |
| 长会议上传 | X-ASR + Silero file VAD | 分段稳定，内存压力小，可边处理边回传 |
| 会后最终稿 | Qwen3-ASR 可选 | 只在 CUDA、依赖、显存都满足时启用 |
| 无 GPU / 演示机 | X-ASR only | 最稳，避免 Qwen 加载失败影响演示 |

建议后续补齐 `480ms / 960ms / 1920ms` profile 后再开放质量档位，否则设置页虽然有多个 profile，但实际只会 fallback 到 160ms，容易让用户误解。

### 3.2 Qwen3-ASR 使用边界

Qwen3-ASR 不应承担实时 partial 识别。建议明确定位为：

- 上传文件最终稿。
- 麦克风停止后的 canonical transcript。
- 不参与 `/ws/live` 的低延迟结果。

建议增加一个“模型能力状态”字段，用于 UI 明确展示：

```json
{
  "qwen3": {
    "available": true,
    "device": "cuda:0",
    "dtype": "bfloat16",
    "loaded": false,
    "estimated_vram_gb": 5.5,
    "mode": "final_transcription_only"
  }
}
```

这样用户能理解：Qwen 可用不等于已加载；已选择 Qwen 不等于实时链路也使用 Qwen。

### 3.3 VAD 策略

当前已有两类 VAD：

- 实时麦克风：`create_live_vad()`，优先 `models/vad/silero_vad.onnx`，缺失时 fallback 到 energy VAD。
- 文件转写：`XASREngine._create_file_segmenter()`，现在也会查 `models/vad/silero_vad.onnx`。

建议统一为一个 VAD 解析模块，例如：

```text
backend/xasr/model_paths.py
  resolve_models_root()
  resolve_xasr_model_dir()
  resolve_qwen3_model_dir()
  resolve_vad_model_path()
  resolve_diarization_model_dir()
```

所有地方只调用这组函数，不再在 `main.py`、`asr_engine.py`、`live_audio.py`、`start.py` 中各自拼路径。

### 3.4 说话人分离模型策略

当前日志显示说话人分离模型缺失时会 fallback 到 ASR-only：

```text
backend/diarization/models/pyannote-segmentation-3.0.int8.onnx
backend/diarization/models/3dspeaker-eres2net.onnx
```

建议把 diarization 模型也迁移到统一模型根目录：

```text
models/diarization/
  pyannote-segmentation-3.0.int8.onnx
  3dspeaker-eres2net.onnx
```

并保持环境变量兼容：

```powershell
$env:DITING_DIARIZATION_SEGMENTATION_MODEL = "models/diarization/pyannote-segmentation-3.0.int8.onnx"
$env:DITING_SPEAKER_EMBEDDING_MODEL = "models/diarization/3dspeaker-eres2net.onnx"
```

优先级建议：

1. 当前比赛/演示如果重点是“转写内容”，说话人分离可以继续作为可选能力。
2. 如果强调“多人会议”，应补齐 diarization 模型并在 UI 显示“说话人分离已启用/已降级”。
3. 不建议在模型缺失时静默显示“Speaker 1/2”，避免误导。

---

## 4. 调试与可观测性优化

### 4.1 增加一键诊断命令

建议新增：

```text
python tools/doctor.py
```

检查内容：

- Python 版本和实际解释器路径。
- `numpy / sherpa_onnx / soundfile / librosa / torch / qwen_asr` 是否安装。
- `models/xasr` 是否存在完整 profile。
- `models/vad/silero_vad.onnx` 是否存在。
- `models/qwen3` 是否包含必要 config、tokenizer、safetensors。
- diarization 模型是否存在。
- 端口 `8765 / 3000` 是否被占用。
- `/api/health` 和 `/api/xasr/status` 是否可访问。

输出示例：

```text
[OK] Python: .venv-qwen3/Scripts/python.exe
[OK] X-ASR low-latency: 586MB
[OK] Silero VAD: models/vad/silero_vad.onnx
[WARN] Diarization: missing models, ASR-only fallback
[WARN] Qwen3: qwen_asr not installed
[OK] Backend port 8765 free
```

这能显著减少“模型实际存在但路径不一致”的排查成本。

### 4.2 标准化日志字段

当前日志能看出问题，但同一事件存在重复 logger 输出、不同模块格式不完全一致。建议对语音链路统一结构化字段：

```text
session_id / file_id
provider
model_dir
vad_provider
segments_count
duration_sec
rtf
fallback_reason
error_code
```

重点日志建议：

- 启动时打印一次最终解析路径：`models_root / xasr / vad / qwen3 / diarization`。
- 每次上传和麦克风最终转写都打印：实际 provider、VAD provider、RTF、是否 fallback。
- Qwen fallback 必须输出具体原因：依赖缺失、模型缺失、CUDA OOM、推理异常、空输出。

### 4.3 前端调试面板

已有日志面板，但建议增加“运行状态”页或设置页块：

- 后端 API revision。
- 当前 ASR provider：selected/effective。
- X-ASR profile：requested/effective/chunk_ms。
- VAD 状态：live/file provider、model path。
- Qwen 状态：依赖、模型路径、device、dtype、loaded。
- Diarization 状态：available/missing models。
- 最近一次录音：received_samples、forwarded_samples、dropped_frames、partial_results、final_results。

这样用户无需看终端就能判断当前降级点。

---

## 5. 后端架构优化

### 5.1 拆分 `backend/main.py`

`backend/main.py` 当前职责过多。建议分阶段拆分：

```text
backend/api/
  health.py
  settings.py
  upload.py
  live.py
  records.py
  summaries.py
  management.py
  logs.py

backend/services/
  transcription_service.py
  live_session_service.py
  upload_service.py
  record_service.py
  summary_service.py
  model_status_service.py
```

短期可以先抽出最容易出错的部分：

1. `/ws/live` → `backend/api/live.py`。
2. `/api/audio/upload` + `/ws/upload/{file_id}` → `backend/api/upload.py`。
3. `_load_xasr_engine()`、`_schedule_xasr_reload()` → `backend/services/model_runtime.py`。
4. 模型路径常量 → `backend/xasr/model_paths.py`。

拆分原则：

- 路由层只解析请求和返回响应。
- service 层负责调用 ASR、VAD、记录、摘要。
- xasr 层不反向依赖 FastAPI。

### 5.2 模型路径集中化

当前模型路径分布在：

- `start.py`
- `backend/main.py`
- `backend/xasr/asr_engine.py`
- `backend/xasr/live_audio.py`
- `backend/xasr/runtime_config.py`

建议新增 `backend/xasr/model_paths.py`：

```python
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def models_root() -> Path:
    return Path(os.getenv("HUIWU_MODELS_DIR", PROJECT_ROOT / "models"))

def xasr_model_dir() -> Path:
    return Path(os.getenv("DITING_XASR_MODEL_DIR", models_root() / "xasr"))

def qwen3_model_dir() -> Path:
    return Path(os.getenv("DITING_QWEN3_MODEL_PATH", models_root() / "qwen3"))

def vad_model_path() -> Path:
    return Path(os.getenv("DITING_SILERO_VAD_PATH", models_root() / "vad" / "silero_vad.onnx"))
```

`start.py` 可以通过轻量导入或复制同一规则，避免再次出现启动器和后端路径不一致。

### 5.3 上传处理任务化

上传长音频当前通过 WebSocket 队列回传进度。建议增强为明确任务对象：

```text
UploadJob
  file_id
  source_path
  status: queued / loading / vad / recognizing / diarizing / saving / complete / error
  progress
  cancel_event
  result
  error
```

收益：

- 前端断线后可重新查询任务状态。
- 支持取消长音频识别。
- 支持后台任务重试。
- 避免 WebSocket 连接本身成为任务生命周期。

### 5.4 资源释放策略

建议建立统一资源释放表：

| 资源 | 当前风险 | 优化建议 |
|---|---|---|
| X-ASR recognizer runtime | 多会话并发时资源占用不可见 | status 输出 active session 数 |
| Qwen3 模型 | CUDA 显存占用大 | 空闲超时卸载、显存水位日志 |
| PROCESSING_EXECUTOR | 长音频阻塞后续任务 | 队列长度、任务超时、取消机制 |
| WebSocket live session | 前端异常断线 | 当前已有 abort，可增加 recover UI |
| recordings | 长期堆积 | 自动清理 `.part` 和过期临时录音 |

---

## 6. 前端架构优化

### 6.1 拆分 `frontend/index.html`

当前 `frontend/index.html` 同时包含：

- 大量 HTML。
- 全量 CSS。
- 会议转写业务。
- 上传逻辑。
- 麦克风 WebSocket 状态机。
- 记录管理。
- 设置页。
- 账号和预约。
- 语音球拖拽。

建议不立即上 React/Vue，而是先做“原生模块化”：

```text
frontend/
  index.html
  css/
    base.css
    layout.css
    transcription.css
    settings.css
    mic-orb.css
  js/
    api-client.js
    websocket-client.js
    transcription-state.js
    upload-controller.js
    live-mic-controller.js
    mic-orb.js
    records-controller.js
    settings-controller.js
    app-init.js
```

短期优先拆：

1. 麦克风相关：`live-mic-controller.js`。
2. 语音球：`mic-orb.js`。
3. 上传：`upload-controller.js`。
4. 设置页：`settings-controller.js`。

原因：这些模块最容易发生异步状态串线。

### 6.2 麦克风状态机显式化

当前麦克风状态隐含在多个变量里：

```js
isRecording
micWs
micAudioContext
micWorklet
micStream
micStopTimer
micSessionToken
micFinalTranscriptReceived
```

建议改成单一状态机：

```js
const MicState = {
  IDLE: 'idle',
  CONNECTING: 'connecting',
  RECORDING: 'recording',
  STOPPING: 'stopping',
  FINALIZING: 'finalizing',
  COMPLETE: 'complete',
  ERROR: 'error',
};
```

所有按钮、语音球、状态栏、WebSocket 回调都只根据这个状态渲染。这样可以避免“按钮显示停止但 WebSocket 已关闭”或“旧会话 cleanup 清掉新会话”的问题。

### 6.3 WebSocket 客户端封装

建议封装 live WebSocket：

```js
class LiveMicClient {
  async connect(settings) {}
  sendFrame(arrayBuffer) {}
  stop() {}
  close(reason) {}
  on(event, handler) {}
}
```

内部处理：

- session token。
- ready/configured 超时。
- stale message guard。
- stop timer。
- close/error 统一回调。

UI 只订阅事件：

```js
client.on('partial', updateLiveText)
client.on('final', appendFinalText)
client.on('finalTranscript', replaceFinalText)
client.on('metrics', renderMetrics)
client.on('error', showError)
```

### 6.4 避免外部 CDN 成为运行风险

当前 WaveSurfer 从 `https://unpkg.com` 动态导入。虽然已有 timeout fallback，但正式演示或内网环境建议：

- 将 WaveSurfer 固定版本下载到 `frontend/vendor/wavesurfer.esm.js`。
- 或完全使用原生 `<audio>` + canvas 占位波形。

比赛现场网络不稳定时，外部 CDN 是不必要风险。

### 6.5 前端测试补强

已有 `frontend/tests/*.test.js`。建议新增：

- 麦克风状态机测试：连续 start/stop/start，不允许旧 session 清理新 session。
- `live-protocol.js` DTP2 帧格式测试。
- 上传 WebSocket fallback 测试：WS 无 segment 时使用 HTTP response。
- 设置 normalize 测试：模型路径、profile fallback、布尔字段。
- 语音球拖拽/dock 状态纯函数测试。

---

## 7. 代码细节优化

### 7.1 Python 代码

建议优先处理：

1. **路径常量重复**  
   统一到 `model_paths.py`。

2. **导入副作用过重**  
   `backend/main.py` import 时会初始化较多对象。建议重资源延后到 lifespan 或 service 初始化阶段，便于测试。

3. **日志函数命名冲突**  
   已修过 `get_recent_logs` 冲突，后续路由函数避免与导入函数同名。

4. **可选依赖检测**  
   `qwen_asr`、`torch`、diarization 模型、WaveSurfer 都应在 status 中明确可用性。

5. **错误码标准化**  
   例如：

   ```text
   MODEL_MISSING
   DEPENDENCY_MISSING
   CUDA_OOM
   VAD_UNAVAILABLE
   WS_PROTOCOL_ERROR
   TRANSCRIPTION_EMPTY
   ```

   前端根据错误码显示可操作建议，而不是只显示 Python 异常字符串。

### 7.2 JavaScript 代码

建议优先处理：

1. **把全局变量收敛到 state 对象**  
   例如 `appState.transcription`、`appState.mic`、`appState.upload`。

2. **禁止多个模块直接改同一 DOM**  
   例如 transcript 渲染应只通过 `renderTranscript()`、`appendSegment()`、`replaceSegments()`。

3. **统一 toast 和 status**  
   错误信息、连接状态、按钮状态应由状态机驱动。

4. **减少 inline style**  
   当前很多渲染片段使用 inline style，建议迁移到 CSS class，减少后续主题维护成本。

5. **音频对象生命周期**  
   WaveSurfer、AudioContext、MediaStream、WebSocket 都应有明确 owner 和 dispose。

### 7.3 测试代码

建议建立三层验证：

```text
unit tests       纯函数、配置、协议、路径解析
integration      ASR engine、VAD、record store、settings
smoke/e2e        start.py、/api/health、/ws/live、上传短音频
```

建议命令：

```powershell
python -m pytest backend/tests -q
node --test frontend/tests/*.test.js
python tools/doctor.py
python backend/tests/smoke_live_websocket.py test_data/wangping.mp3 --url ws://127.0.0.1:8765/ws/live
```

---

## 8. 项目文件夹整理建议

### 8.1 应保留

```text
frontend/
backend/
models/
  xasr/
  vad/
  qwen3/
backend/data/             本地数据库和设置，已 gitignore
backend/logs/             运行日志，已 gitignore
backend/recordings/       麦克风录音，建议 gitignore
```

### 8.2 可归档或清理

| 路径 | 建议 | 原因 |
|---|---|---|
| `_ref_dickrun/` | 确认不再比对后删除或移到项目外 | 参考仓库不应长期混在工作树 |
| `.idea/` | 保持 gitignore，可按个人需要保留 | IDE 本地配置 |
| `test_data/` | 如果测试依赖则保留，否则移到 `tests/fixtures/` | 测试资产位置更清晰 |
| `backend/xasr/zipformer/` | 若只是上游源码参考，建议移到 `third_party/` 或文档说明 | 训练/导出脚本与运行时混杂 |
| `backend/xasr/live_asr.py`、`sherpa_streaming_client.py` | 若不再作为主链路，移入 `tools/` 或 `examples/` | 避免误认为生产入口 |

### 8.3 `.gitignore` 建议补充

确认加入：

```gitignore
models/
download/
backend/recordings/
backend/data/
backend/logs/
*.part
*.wrong_backup
```

如果要保留 `test_data/wangping.mp3` 作为测试样例，应明确白名单或改用小体积 WAV fixture。

---

## 9. 文档优化

当前 `README.md` 和 `TECHNICAL_DOCS.md` 仍有部分旧路径描述，例如 `backend/xasr/models/`。建议同步更新为：

```text
models/xasr/
models/vad/
models/qwen3/
models/diarization/   可选
```

建议文档拆分：

```text
README.md                         产品和快速开始
TECHNICAL_DOCS.md                  当前接口和技术细节
MODEL_SETUP.md                     模型下载、目录、环境变量、常见错误
DEBUGGING.md                       doctor、日志、端口、WebSocket、空转写排查
ARCHITECTURE.md                    前后端架构图和模块边界
PROJECT_OPTIMIZATION_PLAN.md       本优化方案
```

尤其需要新增 `MODEL_SETUP.md`，因为当前最常见问题都集中在模型路径、依赖、profile、VAD、Qwen 运行环境。

---

## 10. 分阶段实施路线

### Phase 0：立即稳定当前可演示版本

目标：不大改架构，只降低再次出错概率。

- 统一 README / TECHNICAL_DOCS 中的模型路径。
- 增加 `tools/doctor.py`。
- `/api/xasr/status` 增加 VAD、Qwen、diarization 的详细缺失原因。
- 前端设置页展示完整模型状态。
- 固化麦克风 start/stop/start 回归测试。
- 将 WaveSurfer 本地化或保留 fallback 并在文档注明。

### Phase 1：模型与调试工程化

目标：让模型切换和故障定位可控。

- 新增 `backend/xasr/model_paths.py`。
- `start.py`、`main.py`、`asr_engine.py`、`live_audio.py`、`runtime_config.py` 全部改用统一路径解析。
- 增加 `MODEL_SETUP.md`。
- 增加模型 profile 自检：显示每个 profile 是否完整。
- 增加 Qwen 空闲卸载和显存日志。
- 增加上传任务状态查询接口。

### Phase 2：前后端拆分

目标：减少单文件复杂度。

- 拆 `frontend/index.html` 中的 CSS 到 `frontend/css/`。
- 拆 live mic、upload、settings、records 到独立 JS 模块。
- 后端抽出 `api/live.py`、`api/upload.py`、`services/model_runtime.py`。
- 为 live mic 引入显式状态机。
- 为上传处理引入 `UploadJob`。

### Phase 3：质量和产品能力提升

目标：提高识别质量、多人会议质量和用户体验。

- 补齐 X-ASR 多 profile 模型，开放 `balanced/meeting/quality` 的真实选择。
- 补齐 diarization 模型并迁移到 `models/diarization/`。
- 增加 diarization 状态和说话人质量指标。
- 优化 VAD 参数预设：短句、会议、访谈、噪声环境。
- 增加历史记录中“原始实时稿 vs 最终稿”的可选查看。
- 增加长音频取消、暂停、重试和后台进度恢复。

---

## 11. 推荐优先级清单

### P0：必须做

- 统一模型路径解析，避免再次出现 `download / models / backend/xasr/models` 混乱。
- 更新 README 和 TECHNICAL_DOCS 的模型目录说明。
- 增加 `doctor.py` 一键诊断。
- 增加麦克风重连回归测试。
- `/api/xasr/status` 输出更完整的模型缺失原因。

### P1：强烈建议

- 拆分前端麦克风和上传逻辑。
- 拆分后端 live/upload 路由。
- 本地化 WaveSurfer 或完全去 CDN 依赖。
- diarization 模型迁移到 `models/diarization`。
- 上传任务对象化，支持断线恢复和取消。

### P2：后续增强

- 补齐 X-ASR 多 profile。
- Qwen3 空闲卸载和显存监控。
- 更细粒度 VAD profile。
- 说话人注册和身份绑定。
- 重叠语音分离。

---

## 12. 验收标准

完成优化后，至少满足以下标准：

1. 新环境运行 `python tools/doctor.py` 能明确指出缺什么模型或依赖。
2. `python start.py` 打印的模型路径与后端实际使用路径完全一致。
3. 设置页能看到 X-ASR、VAD、Qwen、diarization 的真实状态。
4. 麦克风可以连续执行 `开始 → 停止 → 开始 → 停止`，不需要重置页面。
5. 上传短音频和长音频都能显示 provider、VAD、RTF 和 fallback 原因。
6. 没有完整 diarization 模型时，UI 明确显示“ASR-only”，不误导为多人说话人分离已启用。
7. 前端主要业务逻辑不再集中在单个 `index.html` 中。
8. 后端 `main.py` 不再承担全部路由和模型编排。
9. README、TECHNICAL_DOCS、MODEL_SETUP 中的模型路径一致。
10. 全量测试命令有明确说明，并能覆盖核心语音链路。

---

## 13. 结论

当前项目的核心问题不是“功能缺失”，而是“模型路径、异步状态、单文件复杂度和调试入口”还不够工程化。短期应优先做路径集中化、doctor 诊断、文档同步和麦克风/上传回归测试；中期拆分前后端大文件；后期再提升多 profile、Qwen、diarization 和长音频任务能力。

最推荐的下一步是：先实现 `backend/xasr/model_paths.py` 和 `tools/doctor.py`，同时更新 README/TECHNICAL_DOCS 的模型目录。这两项投入小，但能直接解决当前反复出现的启动、模型缺失和调试成本问题。
