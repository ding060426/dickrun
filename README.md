# 谛听 (DiTing) v3.0 — Smart Meeting Speech Cognitive System

基于 X-ASR (sherpa-onnx zipformer2) 的智能会议语音认知系统。支持音频文件上传、实时转写、热词修正、逻辑校验、音频波形可视化。

## 系统架构

```
frontend/          ← 前端 (wavesurfer.js + AudioWorklet + WebSocket)
  audio-worklet.js ← 浏览器采集、真实采样率到 16 kHz 的连续降采样
backend/
  main.py          ← FastAPI 后端 (15 个路由, 4 个 WebSocket)
  xasr/
    asr_engine.py  ← X-ASR 引擎 (VAD + 端点检测 + 转写)
    live_audio.py  ← 实时 PCM 校验 + Silero VAD + ASR 会话边界
    models/        ← ONNX 模型 (需自行下载)
  modules/
    audio_processor.py  ← SNR/RT60 估算 + 热词修正 + 逻辑校验 + 不确定性估计
  utils/
    logger.py      ← 统一日志系统 (控制台 + 文件 + 环形缓冲)
```

## 快速开始

### 1. 环境要求

```bash
pip install fastapi uvicorn numpy
```

### 2. 下载 X-ASR 模型文件

模型文件约 310MB，从 sherpa-onnx 下载：

```bash
mkdir -p backend/xasr/models
cd backend/xasr/models

# 从 ModelScope 下载 (国内推荐)
# 或从 GitHub Releases 下载:
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-06-28.tar.bz2
tar -xjf sherpa-onnx-zipformer-zh-en-2023-06-28.tar.bz2
```

模型目录应有以下文件：
```
backend/xasr/models/
  ├── encoder-160ms.onnx
  ├── decoder-160ms.onnx
  ├── joiner-160ms.onnx
  ├── tokens.txt
  └── silero_vad.onnx       (实时麦克风 VAD，可选；缺失时退化到 ASR 端点)
```

> **可选**：如需说话人分离功能，还需下载声纹模型（~35MB）：
> ```bash
> wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_16k.onnx
> ```

### 3. 获取 Eval_Ali 数据集

数据集中包含远场会议录音和近场单说话人录音，带 TextGrid 标注。

- **来源**: AliMeeting Eval Set (达摩院开源)
- **内容**: 8 场会议, 25 位说话人, 含远场混音和近场纯净音频
- **地址**: [阿里天池 / ModelScope](https://modelscope.cn/datasets) 搜索 `AliMeeting`
- **存放位置**: `Eval_Ali/Eval_Ali/` 目录

目录结构：
```
Eval_Ali/Eval_Ali/
  ├── Eval_Ali_far/
  │   ├── audio_dir/       ← 8 个远场混音 WAV
  │   └── textgrid_dir/    ← 8 个 TextGrid (每场含 4 人标注)
  └── Eval_Ali_near/
      ├── audio_dir/       ← 25 个近场单说话人 WAV
      └── textgrid_dir/    ← 25 个 TextGrid
```

### 4. 启动

```bash
python start.py
```

- 前端: [http://localhost:3000](http://localhost:3000)
- 后端 API: [http://localhost:8765](http://localhost:8765)
- API 文档: [http://localhost:8765/docs](http://localhost:8765/docs)

如果模型未就绪，系统自动进入 Demo 模式，使用预置的模拟会议数据。

### 5. 实时麦克风转写

打开前端后点击顶部 `Mic`，允许浏览器使用麦克风即可。浏览器通过
`AudioWorklet` 连续采集音频，按实际设备采样率降采样为 16 kHz 单声道
`pcm_s16le`，再以带 `DTP2 + uint32 sequence` 头的二进制 WebSocket 帧发送到
`/ws/live`，后端可直接统计丢帧。停止录音时先补齐最后一句，再用落盘 WAV 做一次
完整文件转写，并用这份 canonical transcript 替换低延迟预览结果。录音期间底部会显示由真实
RMS/峰值驱动的彩色声量动画；声量测量只用于本地显示，不会额外上传数据。

当前本地若后端使用非默认端口，可这样打开：

```text
http://localhost:3000/?apiPort=8766
```

实时句尾采用“VAD 候选句尾 + 可撤销宽限期”：Silero 检测到约 500ms
静音后不会立刻重置识别流，而是继续保留上下文；默认再等待 800ms，期间恢复说话就
继续同一句，只有静音持续约 1.3 秒才真正定稿。可通过环境变量调整额外宽限期：

```powershell
$env:DITING_LIVE_ENDPOINT_GRACE_MS = "1000"
python start.py
```

如果仍然容易切出短句，可调到 `1000`～`1400`；如果更重视句尾返回速度，可调到
`400`～`600`。实时 Silero VAD 可用时，sherpa 内置端点会关闭，避免两个端点机制
重复重置解码器；Silero 模型缺失时会回退到依赖无关的流式能量 VAD，而不是把
所有输入都当作语音。直播音频先写入 `backend/recordings/*.wav.part`，正常停止时
原子发布为 `.wav`，异常断线时保留 `.part` 以便恢复。

可选择 `meeting`、`dictation`、`oncall` 三种直播边界档位；模型延迟档位支持
`low-latency`（160ms）、`balanced`（480ms）、`meeting`（960ms）和
`quality`（1920ms），对应 ONNX 文件存在时才可启用：

```powershell
$env:DITING_LIVE_PROFILE = "meeting"
$env:DITING_LIVE_ASR_PROFILE = "low-latency"
$env:DITING_LIVE_VAD = "auto"       # auto / energy
$env:DITING_MAX_UPLOAD_MB = "2048"
$env:DITING_PROCESSING_WORKERS = "2"
python start.py
```

无需物理麦克风也可以用音频文件验证同一条 WebSocket 链路：

```bash
python backend/tests/smoke_live_websocket.py path/to/meeting.wav --url ws://127.0.0.1:8765/ws/live
```

## 功能

| 功能 | 状态 | 说明 |
|------|------|------|
| ASR 转写 | ✅ | sherpa-onnx zipformer2 流式推理 |
| VAD + 端点检测 | ✅ | 文件：能量 VAD；实时：Silero 优先、能量 VAD 降级 + 可配置 pre-roll |
| 热词修正 | ✅ | sherpa modified beam search 原生热词偏置 + 拼音/模糊音后处理 |
| 逻辑校验 | ✅ | 数据冲突检测 (数字/百分比对比) |
| 不确定性估计 | ✅ | 低置信度区段标记 |
| 音频波形可视化 | ✅ | wavesurfer.js + WAV 编码 |
| 实时录音转写 | ✅ | AudioWorklet + DTP2 帧序号 + partial/final + 持久 WAV + 停止后二次定稿 |
| Eval_Ali 评测 | ✅ | CER 计算 / 热词提取 |
| 说话人日志/分段 | 🔜 | LocalMeet 已有 pyannote segmentation + 3D-Speaker 离线实现，尚未迁入本项目 |
| 重叠语音双路分离 | 🔜 | LocalMeet 已有 MossFormer2 离线 GPU 实现；不放入低延迟实时预览主链路 |
| 说话人识别 | 🔜 | 声纹注册库匹配 待实现 |

## License

MIT
