# MODEL_SETUP

会悟默认把本地模型统一放在项目根目录的 `models/` 下。旧路径只作为历史兼容或排查线索，不再作为主模型目录。

## 标准目录

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
    tokenizer.json / tokenizer_config.json
    preprocessor_config.json
    model-*.safetensors
  diarization/
    pyannote-segmentation-3.0.int8.onnx
    3dspeaker-eres2net.onnx
```

## X-ASR profiles

| Profile | Chunk | 用途 | 当前要求 |
|---|---:|---|---|
| `low-latency` | 160ms | 实时麦克风预览 | 当前默认可用 |
| `balanced` | 480ms | 平衡延迟和质量 | 需要补齐对应 ONNX |
| `meeting` | 960ms | 会议文件转写 | 需要补齐对应 ONNX |
| `quality` | 1920ms | 更高质量最终稿 | 需要补齐对应 ONNX |

当前本地只部署 `160ms` 时，真实可用 profile 是 `low-latency`。如果设置页选择其他 profile，后端会 fallback 到已部署 profile，并在 `/api/xasr/status` 中显示 requested/effective。

下载模型默认写入 `models/xasr/`：

```powershell
python backend\xasr\download_models.py --profile low-latency
python backend\xasr\download_models.py --profile meeting
```

## Silero VAD

默认路径：

```text
models/vad/silero_vad.onnx
```

用途：

- 实时麦克风 VAD。
- 上传文件 VAD。
- 说话人分离前的语音段检测。

缺失时：

- 实时链路 fallback 到 energy VAD 或 ASR endpoint。
- 文件转写 fallback 到 whole-file。
- doctor 和 `/api/xasr/status` 会显示缺失路径。

## Qwen3-ASR

默认路径：

```text
models/qwen3/
```

定位：

- 上传文件最终稿。
- 麦克风停止后的 canonical transcript。
- 不参与实时 partial 识别。

额外依赖：

```powershell
.venv-qwen3\Scripts\python.exe -m pip install -r backend\requirements.txt qwen-asr
# torch 按本机 CUDA/CPU 环境单独安装
```

Qwen3 加载失败、依赖缺失或 CUDA OOM 时，后端应回退到 X-ASR，并在 `/api/xasr/status.providers.qwen3` 中显示原因。

## Diarization

推荐路径：

```text
models/diarization/pyannote-segmentation-3.0.int8.onnx
models/diarization/3dspeaker-eres2net.onnx
```

Diarization 是可选能力。模型缺失时系统应明确显示 `ASR-only`，不应误导为说话人分离已启用。

## 环境变量

| 变量 | 含义 | 默认 |
|---|---|---|
| `HUIWU_MODELS_DIR` | 统一模型根目录 | `models/` |
| `DITING_XASR_MODEL_DIR` | X-ASR 模型目录 | `models/xasr/` |
| `DITING_SILERO_VAD_PATH` | Silero VAD ONNX | `models/vad/silero_vad.onnx` |
| `DITING_QWEN3_MODEL_PATH` | Qwen3 模型目录 | `models/qwen3/` |
| `DITING_DIARIZATION_MODEL_DIR` | 说话人分离模型目录 | `models/diarization/` |
| `DITING_DIARIZATION_SEGMENTATION_MODEL` | segmentation ONNX | `models/diarization/pyannote-segmentation-3.0.int8.onnx` |
| `DITING_SPEAKER_EMBEDDING_MODEL` | speaker embedding ONNX | `models/diarization/3dspeaker-eres2net.onnx` |
| `DITING_RECORDINGS_DIR` | 录音保存目录 | `backend/recordings/` |

## 自检

```powershell
python tools\doctor.py
python tools\doctor.py --json
```
