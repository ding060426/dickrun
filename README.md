# 谛听 (DiTing) v3.0 — Smart Meeting Speech Cognitive System

基于 X-ASR (sherpa-onnx zipformer2) 的智能会议语音认知系统。支持音频文件上传、实时麦克风转写、热词修正、逻辑校验、文本后处理（停顿词去除 + MacBERT 同音字纠错）、音频波形可视化。

## 系统架构

```
frontend/              ← 前端 (wavesurfer.js + WebSocket + 麦克风可视化)
  index.html          ← 单页应用 (Vite 构建 / 原生 JS)
backend/
  main.py             ← FastAPI 后端 (15+ 路由, 4 个 WebSocket)
  start.py            ← 一键启动脚本 (后端 + 前端)
  xasr/
    asr_engine.py     ← X-ASR 引擎 (流式推理 + VAD + 端点检测)
    sherpa_streaming_infer.py  ← sherpa-onnx 流式推理封装
    models/           ← ONNX 模型 + tokens.txt (586MB)
    download_models.py ← 模型下载脚本
  modules/
    audio_processor.py       ← SNR/RT60 估算 + 热词修正 + 逻辑校验 + 不确定性估计
    text_postprocessor.py   ← 文本后处理 (Layer 1 规则 + Layer 2 MacBERT 纠错)
  utils/
    logger.py         ← 统一日志系统 (控制台 + 文件 + 环形缓冲)
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 依赖安装：

```bash
pip install fastapi uvicorn[standard] numpy soundfile sherpa-onnx websockets pycorrector pypinyin librosa

# MacBERT 纠错模型依赖 (CPU 版 torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 如果 torch 因 Windows 路径过长安装失败，可安装到短路径：
> `pip install torch --target C:\pylibs`

### 2. 下载 X-ASR 模型文件

模型文件约 586MB，使用项目自带脚本下载：

```bash
cd backend/xasr
python download_models.py
```

模型目录应有以下文件：
```
backend/xasr/models/
  ├── encoder-160ms.onnx    (~295 MB)
  ├── decoder-160ms.onnx    (~2.5 MB)
  ├── joiner-160ms.onnx     (~2.0 MB)
  └── tokens.txt            (~63 KB, 5000 BPE tokens)
```

> **重要**：tokens.txt 必须是 `lang_5000_with_punctuation` 词表（5000 个 BPE token）。
> 错误的词表（如 2002 token 版本）会导致 ASR 输出完全乱码。

### 3. 获取 Eval_Ali 数据集（可选）

数据集中包含远场会议录音和近场单说话人录音，带 TextGrid 标注。

- **来源**: AliMeeting Eval Set (达摩院开源)
- **内容**: 8 场会议, 25 位说话人, 含远场混音和近场纯净音频
- **存放位置**: `dataset/Eval_Ali/` 目录

详见 [数据集使用说明.md](数据集使用说明.md)

### 4. 启动

```bash
python start.py
```

- 前端: http://localhost:3000
- 后端 API: http://localhost:8765
- API 文档: http://localhost:8765/docs

如果模型未就绪，系统自动进入 Demo 模式，使用预置的模拟会议数据。

## 功能

| 功能 | 状态 | 说明 |
|------|------|------|
| ASR 转写 | ✅ | sherpa-onnx zipformer2 流式推理 (CER ~15.7%) |
| VAD + 端点检测 | ✅ | 能量检测 + 动态阈值 + 片段合并 |
| 热词修正 | ✅ | 拼音匹配 + 模糊音校正 |
| 逻辑校验 | ✅ | 数据冲突检测 (数字/百分比对比) |
| 不确定性估计 | ✅ | 低置信度区段标记 |
| 音频波形可视化 | ✅ | wavesurfer.js + WAV 编码 |
| 实时录音转写 | ✅ | WebSocket 流式传输 + 麦克风音量可视化 |
| 文本后处理 | ✅ | Layer 1 规则 (停顿词/重复词) + Layer 2 MacBERT 纠错 |
| Eval_Ali 评测 | ✅ | CER 计算 / 热词提取 |
| 说话人分离 | 🔜 | Embedding + AHC 聚类 待实现 |
| 说话人识别 | 🔜 | 声纹注册库匹配 待实现 |

## 实时麦克风转写

### 工作流程

```
用户点击 Mic
  → 浏览器请求麦克风权限
  → AudioContext (16kHz) + WebSocket 连接 ws://localhost:8765/ws/live
  → 每 256ms 发送一帧音频 (4096 samples, base64 编码)
  → 后端 asyncio.to_thread 调用 sherpa-onnx 流式推理
  → partial 结果实时回传 (文字逐步增长)
  → 端点检测触发 → final 段落 + 文本后处理
  → 前端显示最终文本 + 纠错详情
```

### 文本后处理管道

仅在 **final 段落**（句子结束）时触发，不影响实时性：

```
ASR final 文本
  → Layer 1: 规则后处理 (<1ms)
     - 停顿词去除 (嗯/啊/额/那个/这个/就是...)
     - 重复词合并 (是是是→是, 好的好的→好的)
     - 标点规范化
  → Layer 2: MacBERT 纠错 (~50ms)
     - 同音字纠错 (真→正, 他→她)
     - 形似字纠错
  → 最终文本 (带纠错详情回传前端)
```

## 技术细节

### ASR 模型

- **模型**: X-ASR 160ms streaming zipformer2 transducer (中英文 + 标点)
- **词表**: 5000 BPE tokens (`lang_5000_with_punctuation`)
- **精度**: Eval_Ali 近场 CER 15.7% (准确率 84.3%)
- **推理**: CPU, 1 线程, 实时率 >5x

### WebSocket 端点

| 端点 | 用途 |
|------|------|
| `/ws/live` | 实时麦克风转写 |
| `/ws/upload/{file_id}` | 文件上传进度 |
| `/ws/logs` | 日志流 (调试用) |
| `/ws/meeting` | 演示模式 (模拟会议) |

## 已知问题与修复记录

### ASR 乱码修复 (2026-07-15)

**问题**: ASR 输出完全乱码，CER 100%
**根因**: `models/tokens.txt` 使用了错误的 2002 token 词表（含 256 字节 token），与模型实际使用的 5000 BPE token 词表不匹配
**修复**: 替换为 `lang_5000_with_punctuation/tokens.txt`，CER 降至 15.7%

### 实时麦克风无响应修复 (2026-07-15)

**问题**: 点击 Mic 后浏览器授权麦克风，但说话无任何识别结果
**根因**: 三层 Bug 叠加：
1. 缺少 `websockets` 库，WebSocket 端点返回 404
2. `AudioContext` 未 `resume()`，`onaudioprocess` 不触发
3. `start_session()` 未创建 ASR 实例，`process_chunk` 永远返回空文本
4. `process_chunk` 同步调用阻塞 asyncio 事件循环
5. numpy float32 无法 JSON 序列化，57/73 条消息发送失败

**修复**: 安装 websockets、resume AudioContext、修复 ASR 实例创建、用 asyncio.to_thread、显式类型转换

## License

MIT
