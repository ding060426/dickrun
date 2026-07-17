# DEBUGGING

本文件用于排查会悟本地运行、模型路径、依赖、端口、WebSocket 和上传转写问题。

## 一键诊断

```powershell
python tools\doctor.py
python tools\doctor.py --json
```

检查内容：

- Python 解释器和版本。
- 关键依赖：`numpy`、`sherpa_onnx`、`soundfile`、`librosa`、`torch`、`qwen_asr`。
- 模型：X-ASR profiles、Silero VAD、Qwen3、diarization。
- 端口：`8765`、`3000`。
- 后端接口：`/api/health`、`/api/xasr/status`。

## 启动

```powershell
python start.py
```

启动器会打印后端 Python、模型路径、API docs 和 X-ASR 状态。`start.py` 打印的模型路径应与 `/api/xasr/status.paths` 一致。

## Health 和模型状态

浏览器打开：

```text
http://127.0.0.1:8765/api/health
http://127.0.0.1:8765/api/xasr/status
```

重点看：

- `paths.models_root`
- `paths.xasr_model_dir`
- `paths.vad_model_path`
- `profiles.available_profiles`
- `providers.qwen3.reason`
- `file_vad.available`
- `diarization.mode`
- `resources`

## 常见问题

### X-ASR 模型缺失

现象：

```text
No complete X-ASR profile found
```

检查：

```powershell
python tools\doctor.py
```

确认 `models/xasr/` 中至少有：

```text
encoder-160ms.onnx
decoder-160ms.onnx
joiner-160ms.onnx
tokens.txt
```

### VAD fallback 到 whole-file

现象：

```text
Silero file VAD model missing; file recognition will use whole audio
```

确认：

```text
models/vad/silero_vad.onnx
```

或设置：

```powershell
$env:DITING_SILERO_VAD_PATH = "D:\HACKERMarathon\Project\Huiwu\models\vad\silero_vad.onnx"
```

### Qwen3 不可用

Qwen3 是可选最终稿能力，不影响 X-ASR 实时 partial。

常见原因：

- `torch` 未安装。
- `qwen_asr` 未安装。
- `models/qwen3/` 缺 `config.json` 或 `.safetensors`。
- CUDA OOM。

状态位置：

```text
/api/xasr/status -> providers.qwen3
```

### Diarization 显示 ASR-only

原因：

```text
models/diarization/pyannote-segmentation-3.0.int8.onnx
models/diarization/3dspeaker-eres2net.onnx
```

缺任意一个都会 fallback 到 ASR-only。这是可控降级，不是上传或转写失败。

### 上传没有进度或结果

检查：

1. 浏览器 Network 中 `/ws/upload/{file_id}` 是否连接。
2. `/api/audio/upload` 是否返回 200。
3. 后端日志是否出现 `Upload:` 和 ASR processing。
4. `/api/xasr/status.file_vad` 是否可用。
5. 如果 WebSocket 没有 segment，前端应使用 HTTP response fallback。

### 麦克风停止后再次开始无结果

应覆盖的行为：

- 前端有 session token guard。
- 旧 WebSocket message 不得清理新 session。
- `AudioWorklet`、`MediaStream`、`WebSocket` cleanup 必须带 session token。

回归测试目标：

```text
开始 -> 停止 -> 开始 -> 停止
```

无需刷新页面。

## 日志位置

```text
backend/logs/diting.log
backend/logs/errors.log
```

建议排查字段：

- `session_id` / `file_id`
- `provider`
- `model_dir`
- `vad_provider`
- `segments_count`
- `duration_sec`
- `rtf`
- `fallback_reason`
- `error_code`

## 测试命令

```powershell
python -m unittest backend.tests.test_model_paths backend.tests.test_runtime_config backend.tests.test_engine_pool backend.tests.test_startup_launcher backend.tests.test_doctor backend.tests.test_model_status_service
node --test frontend\tests\*.test.js
python backend\tests\smoke_live_websocket.py test_data\wangping.mp3 --url ws://127.0.0.1:8765/ws/live
```
