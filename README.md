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
`pcm_s16le`，再以二进制 WebSocket 帧发送到 `/ws/live`。停止录音时会等待
最后一句完成后再关闭连接，不会丢弃尾部文字。

当前本地若后端使用非默认端口，可这样打开：

```text
http://localhost:3000/?apiPort=8766
```

无需物理麦克风也可以用音频文件验证同一条 WebSocket 链路：

```bash
python backend/tests/smoke_live_websocket.py path/to/meeting.wav --url ws://127.0.0.1:8765/ws/live
```

## 功能

| 功能 | 状态 | 说明 |
|------|------|------|
| ASR 转写 | ✅ | sherpa-onnx zipformer2 流式推理 |
| VAD + 端点检测 | ✅ | 文件：能量 VAD；实时麦克风：sherpa-onnx Silero VAD + 200ms pre-roll |
| 热词修正 | ✅ | 拼音匹配 + 模糊音校正 |
| 逻辑校验 | ✅ | 数据冲突检测 (数字/百分比对比) |
| 不确定性估计 | ✅ | 低置信度区段标记 |
| 音频波形可视化 | ✅ | wavesurfer.js + WAV 编码 |
| 实时录音转写 | ✅ | AudioWorklet 连续采集 + 16 kHz 二进制 PCM + partial/final |
| Eval_Ali 评测 | ✅ | CER 计算 / 热词提取 |
| 说话人日志/分段 | 🔜 | LocalMeet 已有 pyannote segmentation + 3D-Speaker 离线实现，尚未迁入本项目 |
| 重叠语音双路分离 | 🔜 | LocalMeet 已有 MossFormer2 离线 GPU 实现；不放入低延迟实时预览主链路 |
| 说话人识别 | 🔜 | 声纹注册库匹配 待实现 |

## License

MIT
