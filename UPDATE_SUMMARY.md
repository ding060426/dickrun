# 会悟项目更新总结

本文记录从合并 `main` 分支以来，本分支完成的主要修复、迁移、优化和验收结果。

## 1. 合并目标

本轮工作的初始目标是：

- 以后端和主体前端结构尽量对齐 `main` 分支。
- 只保留当前分支中前端麦克风输入语音球功能，包括：
  - 录音时显示；
  - 音量动态反馈；
  - 拖动；
  - 贴边隐藏；
  - 点击恢复。
- 修复上传转写、模型路径、VAD、麦克风重复启动等问题。
- 在后续优化阶段中，把模型路径、调试入口、前后端结构和文档体系工程化。

## 2. Git 与分支处理

已确认远端仓库：

```text
https://github.com/ding060426/dickrun.git
```

当前工作从 `main` 体系恢复后继续在本地分支上推进，并准备推送到远端 `stucture-update` 分支。

## 3. 模型目录迁移与路径修复

### 3.1 统一模型目录

项目模型目录已统一为项目根目录下的 `models/`：

```text
models/
  xasr/
    encoder-160ms.onnx
    decoder-160ms.onnx
    joiner-160ms.onnx
    tokens.txt
  vad/
    silero_vad.onnx
  qwen3/
    config.json
    model-*.safetensors
  diarization/
    pyannote-segmentation-3.0.int8.onnx
    3dspeaker-eres2net.onnx
```

当前真实部署的 X-ASR profile 是 `low-latency` / `160ms`。`balanced`、`meeting`、`quality` 只有补齐对应 ONNX 后才会显示为完整可用。

### 3.2 新增集中路径解析

新增：

- `backend/xasr/model_paths.py`

集中处理：

- 项目根目录；
- 模型根目录；
- X-ASR 模型目录；
- Qwen3 模型目录；
- Silero VAD 路径；
- diarization 模型路径；
- recordings、settings、hotwords 路径；
- X-ASR profile 文件完整性检查。

已替换路径分散点：

- `start.py`
- `backend/main.py`
- `backend/xasr/asr_engine.py`
- `backend/xasr/live_audio.py`
- `backend/xasr/runtime_config.py`
- `backend/xasr/engine_pool.py`
- `backend/xasr/download_models.py`

### 3.3 修复旧路径问题

修复了此前后端误查旧目录的问题：

```text
backend/xasr/models/
```

现在启动器和后端实际使用路径保持一致：

```text
models/xasr/
models/vad/silero_vad.onnx
models/qwen3/
```

## 4. 上传转写与 VAD 修复

### 4.1 上传转写模型路径修复

此前上传时后端报：

```text
No complete X-ASR profile found in backend/xasr/models
X-ASR model is not loaded; transcription cannot start
```

原因是模型实际已迁移到 `models/xasr/`，但部分代码仍使用旧路径。

已通过 `model_paths.py` 和相关调用方替换修复。

### 4.2 文件 VAD 修复

此前文件识别 VAD 报：

```text
Silero file VAD model missing; file recognition will use whole audio
```

原因是 `asr_engine.py` 只查 `models/xasr/silero_vad.onnx`，而实际 VAD 在：

```text
models/vad/silero_vad.onnx
```

已修复为统一调用 `resolve_vad_model_path()`。

## 5. 麦克风重复启动修复

修复了麦克风识别在“停止后再次开始”时不再显示识别内容的问题。

原因是旧 WebSocket 的 `stopped` / `final_transcript` 消息可能晚到，旧会话 cleanup 会清掉新会话的 `micWs`、`AudioContext`、`AudioWorklet` 等资源。

修复内容：

- 增加 `micSessionToken`；
- WebSocket message handler 增加 stale guard；
- stop timer 增加 token guard；
- `cleanupMicRecording(sessionToken)` 增加 token guard；
- cleanup 时断开旧 `AudioWorkletNode` 并清空 `port.onmessage`。

## 6. Doctor 诊断工具

新增：

- `tools/doctor.py`
- `backend/tests/test_doctor.py`

支持：

```powershell
python tools/doctor.py
python tools/doctor.py --json
```

诊断内容：

- Python 解释器和版本；
- 关键依赖：`numpy`、`sherpa_onnx`、`soundfile`、`librosa`、`torch`、`qwen_asr`；
- X-ASR profile 完整性；
- Silero VAD；
- Qwen3 模型文件；
- diarization 模型；
- 端口 `8765` / `3000`；
- 后端 `/api/health` 和 `/api/xasr/status`。

## 7. `/api/xasr/status` 状态增强

新增：

- `backend/services/model_status_service.py`
- `backend/tests/test_model_status_service.py`

`/api/xasr/status` 现在统一输出：

- `paths`：模型和运行路径；
- `profiles`：requested/effective profile、fallback、各 profile 文件完整性；
- `providers`：X-ASR / Qwen3 provider 状态；
- `file_vad`：文件 VAD provider、阈值、模型路径；
- `live_vad`：实时 VAD provider、endpoint policy；
- `diarization`：可用性、ASR-only fallback、缺失模型；
- `resources`：推理线程、上传任务、active sessions 等；
- `features`：热词、逻辑校验、不确定性、说话人分离等能力。

保留旧字段，避免破坏前端兼容。

## 8. 后端结构优化

### 8.1 新增 service 层

新增：

- `backend/services/__init__.py`
- `backend/services/model_runtime.py`
- `backend/services/model_status_service.py`
- `backend/services/upload_service.py`
- `backend/services/live_session_service.py`

完成内容：

- `ModelRuntimeService` 接管模型加载、reload coalesce、hotword 配置、runtime close；
- `UploadJob` / `UploadJobStore` 管理上传任务状态、进度、订阅、取消标记；
- `model_status_service` 统一构建模型状态 payload。

### 8.2 API 边界文件

新增：

- `backend/api/__init__.py`
- `backend/api/upload.py`
- `backend/api/live.py`

当前 `upload/live` 主路由仍保留在 `backend/main.py`，原因是它们仍依赖大量转换函数、pipeline、registry、executor 和 WebSocket cleanup。已建立 API 边界，为后续机械迁移做准备。

### 8.3 上传任务化

新增接口：

```text
GET  /api/audio/upload/{file_id}/status
POST /api/audio/upload/{file_id}/cancel
```

WebSocket 断开不再等同于任务取消，上传任务状态可查询。

## 9. Qwen3 资源释放增强

修改：

- `backend/xasr/qwen3_engine.py`
- `backend/xasr/engine_pool.py`

新增：

- `runtime_status()`；
- `maybe_unload_idle()`；
- `last_used_at`；
- `idle_timeout_sec`；
- `device`；
- `dtype`；
- `estimated_vram_gb`；
- `last_error`。

新增测试：

- `backend/tests/test_qwen3_idle_unload.py`

Qwen3 仍只用于上传和最终稿，不参与实时 partial。

## 10. 前端结构优化

### 10.1 CSS 拆分

`frontend/index.html` 中原本的大段 inline CSS 已拆分为：

- `frontend/css/base.css`
- `frontend/css/transcription.css`
- `frontend/css/mic-orb.css`
- `frontend/css/layout.css`
- `frontend/css/settings.css`

`index.html` 改为通过 `<link>` 引入。

### 10.2 麦克风和语音球模块支点

新增：

- `frontend/js/live-mic-client.js`
- `frontend/js/live-mic-controller.js`
- `frontend/js/mic-orb.js`

新增测试：

- `frontend/tests/live-mic-client.test.js`
- `frontend/tests/live-mic-controller.test.js`
- `frontend/tests/mic-orb.test.js`

覆盖：

- DTP2 frame 发送；
- ready/configured 流程；
- stop 发送；
- 麦克风状态机；
- start/stop/start 防旧会话 cleanup；
- 语音球位置 clamp 和贴边判断。

### 10.3 API / Upload / Settings / Records 模块支点

新增：

- `frontend/js/api-client.js`
- `frontend/js/upload-controller.js`
- `frontend/js/settings-controller.js`
- `frontend/js/records-controller.js`
- `frontend/js/app-init.js`

新增测试：

- `frontend/tests/api-client.test.js`
- `frontend/tests/upload-controller.test.js`
- `frontend/tests/settings-controller.test.js`
- `frontend/tests/records-controller.test.js`

当前这些模块作为低风险支点存在，未一次性搬空 `index.html` 的全部业务逻辑，避免破坏现有演示链路。

### 10.4 WaveSurfer 风险处理

新增：

- `frontend/vendor/wavesurfer.esm.js`

加载顺序调整为：

1. 本地 vendor；
2. CDN；
3. 原生 `<audio>` + placeholder fallback。

即使 WaveSurfer 不可用，历史/上传音频播放仍可 fallback。

## 11. 文档更新

新增：

- `PROJECT_OPTIMIZATION_PLAN.md`
- `MODEL_SETUP.md`
- `DEBUGGING.md`
- `ARCHITECTURE.md`
- `UPDATE_SUMMARY.md`

更新：

- `README.md`
- `TECHNICAL_DOCS.md`

已修复旧文档问题：

- 不再把 `backend/xasr/models/` 作为主模型目录；
- 前端测试命令改为 `node --test frontend/tests/*.test.js`；
- 增加 `tools/doctor.py` 诊断说明；
- 明确 Qwen3 只用于最终稿；
- 明确 diarization 缺模型时为 ASR-only fallback；
- 明确高风险目录暂不删除。

## 12. 高风险目录处理策略

已按用户要求“先不删除”。以下目录未删除、未移动：

- `_ref_dickrun/`
- `.idea/`
- `backend/xasr/zipformer/`
- `backend/recordings/`
- `models/`
- `backend/data/`
- `backend/logs/`

只通过 `.gitignore`、文档和 doctor 说明管理。

## 13. 验收结果

### 13.1 后端测试

```text
Ran 25 tests in 5.340s
OK
```

覆盖：

- model paths；
- runtime config；
- engine pool；
- startup launcher；
- doctor；
- model status；
- model runtime；
- upload job；
- Qwen3 idle unload。

### 13.2 前端测试

```text
# tests 21
# pass 21
# fail 0
```

覆盖：

- app settings；
- audio worklet；
- live protocol；
- live mic client；
- live mic controller；
- mic orb；
- api client；
- upload controller；
- settings controller；
- records controller；
- page runtime smoke；
- management page integration；
- record summary；
- presentation。

### 13.3 Doctor 输出摘要

```text
[OK] X-ASR low-latency: 160ms
[OK] Silero VAD: models/vad/silero_vad.onnx
[OK] Qwen3 model: models/qwen3
[WARN] Diarization: missing 2 file(s); ASR-only fallback
[OK] Port 8765: free
[OK] Port 3000: free
```

当前 WARN：

- 当前 `venv` 缺少 `numpy`、`sherpa_onnx`、`soundfile`、`librosa`、`torch`、`qwen_asr`；
- 后端未启动时 `/api/health` 和 `/api/xasr/status` 会显示 API WARN；
- `models/diarization/` 缺两个可选模型，按计划 ASR-only fallback。

## 14. 仍保留的后续可优化项

当前已经完成安全拆分和工程化支点，但仍未强行完成以下高风险机械迁移：

- 将 `/api/audio/upload` 和 `/ws/live` 主体完全移出 `backend/main.py`；
- 将 `frontend/index.html` 的全部业务 JS 完全搬空到 controller 文件；
- 下载并替换真实 `frontend/vendor/wavesurfer.esm.js`；
- 补齐 `models/diarization/` 真实模型；
- 补齐 X-ASR `480ms / 960ms / 1920ms` 多 profile 模型。

保留这些项是为了避免一次性大拆导致演示主链路回归风险。当前已建立 service/API/frontend 模块边界和测试支点，后续可按小 PR 继续迁移。
