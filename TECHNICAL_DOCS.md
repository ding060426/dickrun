# 谛听 (DiTing) v3.0 — 技术文档

> **Smart Meeting Speech Cognitive System**
> 基于 X-ASR (sherpa-onnx zipformer2) 的智能会议语音认知系统

---

## 目录

1. [系统概览](#1-系统概览)
2. [技术栈](#2-技术栈)
3. [项目结构](#3-项目结构)
4. [AI 模型清单](#4-ai-模型清单)
5. [核心参数配置](#5-核心参数配置)
6. [处理管线详解](#6-处理管线详解)
7. [API 接口文档](#7-api-接口文档)
8. [WebSocket 协议](#8-websocket-协议)
9. [前端架构](#9-前端架构)
10. [部署指南](#10-部署指南)
11. [开发指南](#11-开发指南)
12. [已知限制与改进方向](#12-已知限制与改进方向)

---

## 1. 系统概览

DiTing v3.0 是一个面向真实会议场景的端到端语音转写与认知增强系统。用户上传会议音频（mp3/wav/flac），系统自动完成：

```
原始音频 → VAD 语音检测 → X-ASR 流式转写 → 文本后处理 → 波形可视化
                                    ↓
                         热词修正 + 逻辑校验 + 不确定性估计
```

**核心指标（实测）：**

| 指标 | 数值 |
|------|------|
| 模型总大小 (ASR+Punct) | ~585 MB |
| ASR 推理速度 (RTF) | ~0.003 (CPU, 4线程) |
| VAD 切句精度 | 300ms 停顿阈值 |
| 标点恢复延迟 | <50ms/句 |
| 支持音频格式 | mp3, wav, flac, m4a, ogg |
| 最大处理时长 | 无硬限制（实测 ~30min 会议正常） |

---

## 2. 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 主语言 |
| FastAPI | ≥0.104 | Web 框架 + WebSocket |
| Uvicorn | ≥0.24 | ASGI 服务器 |
| NumPy | ≥1.24 | 数组运算 |
| SciPy | ≥1.11 | 信号处理 |
| librosa | ≥0.10 | 音频解码 |
| soundfile | ≥0.12 | WAV 读写 |
| sherpa-onnx | ≥1.12 | ONNX 推理引擎 (ASR + 标点) |
| python-multipart | ≥0.06 | 文件上传解析 |

### 前端

| 技术 | 用途 |
|------|------|
| 原生 HTML5 + CSS3 | 无框架依赖，纯原生 |
| ES Module | 脚本加载 |
| wavesurfer.js v7.8 | 音频波形渲染 (CDN) |
| WebSocket API | 实时数据推送 |
| Fetch API | HTTP 通信 |

### 推理引擎

| 引擎 | 用途 |
|------|------|
| ONNX Runtime | sherpa-onnx 内置，CPU 推理 |
| Zipformer2 | 流式 Transducer 编码器 |
| CT-Transformer | 标点恢复 (阿里巴巴达摩院) |

---

## 3. 项目结构

```
dickrun/
├── start.py                          # 一键启动脚本
├── README.md                         # 快速入门
├── .gitignore                        # 排除大数据文件
│
├── backend/                          # ====== 后端 ======
│   ├── requirements.txt              # Python 依赖
│   ├── main.py                       # FastAPI 应用入口 (864行)
│   │   ├── 15 个 HTTP 路由
│   │   ├── 4  个 WebSocket 端点
│   │   └── WAV 编码器 + 上传管线
│   │
│   ├── modules/                      # 认知增强模块
│   │   ├── __init__.py               # 模块导出
│   │   ├── audio_processor.py        # SNR/RT60/热词/逻辑/不确定性 (538行)
│   │   ├── text_post_processor.py    # 文本后处理管线 (343行)
│   │   └── punctuation_model.py      # ML标点恢复包装器 (259行)
│   │
│   ├── utils/                        # 工具
│   │   └── logger.py                 # 统一日志 (187行)
│   │
│   ├── xasr/                         # X-ASR 引擎
│   │   ├── asr_engine.py             # 核心引擎 (804行)
│   │   │   ├── _energy_vad()         # 能量VAD
│   │   │   ├── XASREngine            # 引擎主类
│   │   │   └── _build_result()       # 结果构建 + 后处理集成
│   │   ├── sherpa_streaming_infer.py # sherpa-onnx 流式包装 (171行)
│   │   ├── live_asr.py               # 实时录音ASR
│   │   ├── download_models.py        # 模型下载器
│   │   ├── models/                   # ONNX 模型目录
│   │   │   ├── encoder-160ms.onnx    # Zipformer2 编码器 (~295MB)
│   │   │   ├── decoder-160ms.onnx    # 解码器 (~2.5MB)
│   │   │   ├── joiner-160ms.onnx     # 连接器 (~2.0MB)
│   │   │   ├── tokens.txt            # BPE 词表 (2002 tokens)
│   │   │   └── punct/                # 标点恢复模型
│   │   │       ├── model.onnx        # CT-Transformer (~281MB)
│   │   │       └── tokens.json       # token→id 映射
│   │   └── zipformer/                # Zipformer 训练管线 (28个文件)
│   │
│   ├── eval_ali_integration.py       # Eval_Ali 数据集集成 (447行)
│   └── run_demo.py                   # 离线命令行处理工具
│
├── frontend/                         # ====== 前端 ======
│   └── index.html                    # 单文件 SPA (1696行)
│       ├── AudioPlaybackManager      # wavesurfer.js 管理器
│       ├── 上传管线 + 回退策略
│       └── 实时 WebSocket 渲染
│
└── Eval_Ali/                         # ====== 数据集 (不提交Git) ======
    └── Eval_Ali/
        ├── Eval_Ali_far/             # 8场远场混音会议
        └── Eval_Ali_near/            # 25位说话人近场音频
```

---

## 4. AI 模型清单

### 4.1 主 ASR 模型：sherpa-onnx Zipformer2

| 属性 | 值 |
|------|-----|
| 架构 | Zipformer2 Transducer (流式) |
| 编码器 | `encoder-160ms.onnx` — 12层 Zipformer, 80维 FBank 输入 |
| 解码器 | `decoder-160ms.onnx` — 无状态解码器, 2002 token 词表 |
| 连接器 | `joiner-160ms.onnx` — 声学+语言联合网络 |
| 词表 | BPE (Byte Pair Encoding), 2002 tokens, 不含标点 |
| 语言 | 中文 + 英文混合 |
| 采样率 | 16000 Hz, 单声道, 16-bit |
| 帧移 | 10ms |
| 块大小 | 160ms (= 2560 samples) |
| 解码策略 | Greedy Search (贪心搜索) |
| 推理后端 | ONNX Runtime, CPU (`provider="cpu"`) |
| 总大小 | ~300 MB |
| 来源 | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) |

**关键行为：**
- **流式解码**：音频以 200ms 块输入，实时输出部分识别结果
- **端点检测**：基于 blank frame 计数 (`enable_endpoint_detection=True`)
- **无标点输出**：模型训练时不包含标点 token，标点在后续由专用模型恢复

### 4.2 标点恢复模型：CT-Transformer

| 属性 | 值 |
|------|-----|
| 架构 | CT-Transformer (Classification-Tagging Transformer) |
| 模型文件 | `punct/model.onnx` |
| 词汇表 | `punct/tokens.json` — 272,727 tokens |
| 语言 | 中文 + 英文 |
| 标点类型 | ， 。 ！ ？ 、 ： ； |
| 推理后端 | ONNX Runtime, CPU |
| 大小 | ~281 MB |
| 推理速度 | <50ms/句 |
| 来源 | [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) |
| 研发方 | 阿里巴巴达摩院 |

**关键行为：**
- 输入无标点文本 → 输出带完整中文标点的文本
- 支持 4 种标点：逗号、句号、问号、感叹号
- 模型自动判断断句位置，无需预设规则

### 4.3 VAD（语音活动检测）

| 属性 | 值 |
|------|-----|
| 类型 | 基于短时能量 (Short-Time Energy) |
| 算法 | RMS 能量 + 动态阈值 + 凝聚合并 |
| 框架长度 | 25ms |
| 跳跃长度 | 10ms |

---

## 5. 核心参数配置

### 5.1 VAD 参数 (`_energy_vad`)

| 参数 | 默认值 | 含义 | 调优建议 |
|------|--------|------|----------|
| `energy_threshold_ratio` | **0.06** | 能量阈值（相对于最大能量） | 嘈杂环境提高至 0.08-0.10 |
| `min_speech_frames` | **15** | 最小语音帧数 (~150ms) | 检测更短/更长的最小语音 |
| `min_silence_frames` | **30** | 最小静音帧数 (~300ms) | **控制断句粒度**：越小句子越短 |
| `pre_padding_ms` | **200** | 语音前扩展 | 防止切掉句首辅音 |
| `post_padding_ms` | **200** | 语音后扩展 | 防止切掉句尾 |
| `min_segment_duration` | **0.8s** | 最小段长 | 更小的值允许独立短句 |
| `frame_ms` | 25 | 分析帧长 | 不变 |
| `hop_ms` | 10 | 帧移 | 不变 |

**影响断句的关键参数：`min_silence_frames`**

```
min_silence_frames=50 (旧值 500ms):  "我们讨论了预算然后决定了下周再审一下产品策略"
min_silence_frames=30 (新值 300ms):  "我们讨论了预算" "然后决定了下周再审一下" "产品策略"
```

### 5.2 XASREngine 构造函数

```python
XASREngine(
    hotwords: List[str] = None,              # 领域热词列表
    enable_logic_validation: bool = True,     # 逻辑冲突检测
    enable_hotword_correction: bool = True,   # 拼音模糊匹配修正
    enable_uncertainty: bool = True,          # 低置信度标记
    enable_endpoint_detection: bool = True,   # sherpa-onnx 端点检测
    enable_text_postprocess: bool = True,     # 【新增】文本后处理管线
    model_dir: str = None,                    # 模型目录
    provider: str = "cpu",                    # ONNX 后端: cpu/cuda/coreml
    sample_rate: int = 16000,                 # 采样率
    num_threads: int = 2,                     # 推理线程数
    decoding_method: str = "greedy_search",   # 解码策略
)
```

### 5.3 文本后处理配置

```python
process_asr_text(
    raw_text: str,
    enable_filler_filter: bool = True,   # 语气词过滤
    enable_punctuation: bool = True,     # ML标点恢复 (模型优先)
    enable_force_split: bool = True,     # 长句强制断句
    enable_normalize: bool = True,       # 规范化
    max_chars_per_segment: int = 40,     # 单句最大汉字数
)
```

### 5.4 上传与流式处理

| 参数 | 值 | 含义 |
|------|-----|------|
| 音频块大小 | 200ms | 每次送入 ASR 的音频长度 |
| 块重叠 | 50ms | 防止边界信息丢失 |
| WebSocket 超时 | 600s | 处理超时 (10分钟) |
| HTTP 回退计时器 | 30s | WS 无响应的等待时长 |
| 最大 WaveSurfer 实例 | 40 | LRU 淘汰旧波形 |

---

## 6. 处理管线详解

### 6.1 音频上传完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 前端                          │ 后端                              │
├───────────────────────────────┼───────────────────────────────────┤
│ 1. 用户选择文件               │                                   │
│ 2. 生成 file_id (UUID)        │                                   │
│ 3. 打开 WS /ws/upload/{id}   │ 4. 注册 upload_sessions[id]=Queue │
│ 5. POST /api/audio/upload    │                                   │
│    ?file_id={id}              │ 6. 匹配 Queue → 流式模式          │
│                               │ 7. 保存临时文件                    │
│                               │ 8. librosa 加载 (16kHz 重采样)    │
│                               │ 9. _energy_vad() → 语音片段列表   │
│                               │ 10. 逐片段处理 (线程池):           │
│                               │     a. estimate_snr / rt60        │
│                               │     b. SherpaStreamingASR.create  │
│                               │     c. 200ms块送入 accept_waveform│
│                               │     d. decode() → 部分结果        │
│                               │     e. is_endpoint()? → 截断     │
│                               │     f. _numpy_to_wav_base64()     │
│                               │ 11. 每段结果 WS 推送 → 前端       │
│ 12. 渲染 segment 卡片         │                                   │
│     + wavesurfer 波形         │                                   │
│     + 文本显示                 │                                   │
│     + 自动播放前3段            │                                   │
│ 13. WS 'complete' → 结束      │ 14. 清理临时文件                   │
└───────────────────────────────┴───────────────────────────────────┘
```

### 6.2 文本后处理管线

```
原始 ASR 文本 (无标点, 有语气词)
    │
    ├─ ① 语气词过滤 (remove_fillers)
    │     ┌─────────────────────────────────────┐
    │     │ 句首删除: 嗯, 呃, 欸, 嘶, 啧...     │
    │     │ 句尾删除: 啊, 哦, 嘛, 呗, 吧...     │
    │     │ 句中压缩: "对对对"→"对的"             │
    │     │          "是是是"→"是的"             │
    │     │          "就是说呢"→""               │
    │     └─────────────────────────────────────┘
    │
    ├─ ② 标点恢复 (restore_punctuation)
    │     ┌─────────────────────────────────────┐
    │     │ 优先: CT-Transformer (281MB ONNX)   │
    │     │   输入: "今天讨论预算问题"           │
    │     │   输出: "今天讨论预算问题。"         │
    │     │                                     │
    │     │ 兜底: 规则引擎                       │
    │     │   "但是/所以/然后" → 前插句号       │
    │     │   "吗/呢/吧"结尾 → 问号             │
    │     └─────────────────────────────────────┘
    │
    ├─ ③ 强制断句 (force_split_long_sentence)
    │     ┌─────────────────────────────────────┐
    │     │ 单句超过40个汉字且无标点            │
    │     │ → 在"的/了/和/与/及"后插入逗号     │
    │     └─────────────────────────────────────┘
    │
    └─ ④ 规范化 (normalize_text)
          ┌─────────────────────────────────────┐
          │ 连续重复字3次+ → 压缩为2次          │
          │ 中英文数字间加空格                  │
          │ 清理多余空格                        │
          └─────────────────────────────────────┘
    │
    ▼
干净、有标点、合理长度的文本
```

### 6.3 WAV 编码流程

```
float32 numpy 数组 ([-1, 1])
    │
    ├─ 归一化: audio = audio / max(|audio|)
    ├─ 量化: int16 = clip(float32 * 32767, -32768, 32767)
    ├─ 构建 44 字节 RIFF/WAV 头:
    │     RIFF头 + fmt块 (PCM=1, mono, 16kHz, 16bit) + data块
    ├─ 拼接: header + int16.tobytes()
    └─ Base64 编码 → ASCII 字符串
    │
    ▼
通过 WebSocket JSON 发送到前端
前端: atob() → Uint8Array → Blob → wavesurfer.loadBlob()
```

---

## 7. API 接口文档

### 7.1 HTTP 路由一览

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 (X-ASR 状态、模型加载状态) |
| `GET` | `/api/xasr/status` | X-ASR 详细状态 (模型路径、功能开关、热词数) |
| `GET` | `/api/meeting/demo` | 获取演示会议数据 (2个预设片段) |
| `GET` | `/api/hotwords` | 获取热词、逐词权重和模糊拼音设置 |
| `POST` | `/api/hotwords` | 兼容接口：向现有配置追加热词 (`{"words": [...]}`) |
| `PUT` | `/api/hotwords` | 替换并持久化完整热词设置；新识别会话生效 |
| `POST` | `/api/audio/upload` | **上传音频** (multipart, `?file_id=UUID`) |
| `GET` | `/api/logs/recent` | 获取最近 N 条日志 (`?n=50`) |
| `GET` | `/api/logs/download` | 下载完整日志文件 |
| `GET` | `/api/eval/status` | Eval_Ali 数据集状态 |
| `GET` | `/api/eval/hotwords` | 从数据集中提取领域热词 |
| `GET` | `/api/eval/meeting/{id}` | 获取特定会议的 TextGrid 标注 |
| `GET` | `/app` | SPA 入口 |

### 7.2 关键接口详情

**POST /api/audio/upload**

```http
POST /api/audio/upload?file_id=550e8400-e29b-41d4-a716-446655440000
Content-Type: multipart/form-data

file: meeting.mp3
```

响应 (`status: processing`, WS 流式模式):
```json
{
  "file_id": "550e8400-...",
  "filename": "meeting.mp3",
  "status": "processing",
  "size_mb": 27.2,
  "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)"
}
```

响应 (`status: completed`, 同步模式, 无 WS 时):
```json
{
  "file_id": "...",
  "filename": "meeting.mp3",
  "status": "completed",
  "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
  "segments_count": 146,
  "segments": [
    {
      "index": 1,
      "text": "今天我们来讨论一下产品的转化率问题。",
      "raw_text": "今天我们来讨论一下产品的转化率问题",
      "start_sec": 0.0,
      "end_sec": 5.2,
      "asr_confidence": 0.85,
      "snr_db": 28.0,
      "rt60": 0.3,
      "quality_score": 0.85,
      "quality_label": "high",
      "corrections": [{"position": 21, "original": "bat", "corrected": "BERT", "method": "pinyin_match"}],
      "logic_flags": [],
      "terms": ["转化率"],
      "data_points": [],
      "uncertain_spans": [],
      "uncertainty": {},
      "audio_wav_base64": "UklGRiR/BQBXQVZFZm10..."
    }
  ]
}
```

### 7.3 错误状态码

| status | 含义 |
|--------|------|
| `processing` | 文件已接收，WS 流式输出中 |
| `completed` | 处理完成 |
| `demo_mode` | X-ASR 模型未就绪，返回 Demo 数据 |
| `error` | 处理失败 |

---

## 8. WebSocket 协议

### 8.1 端点列表

| 端点 | 用途 | 方向 |
|------|------|------|
| `/ws/upload/{file_id}` | 上传处理进度 | 后端 → 前端 |
| `/ws/meeting` | Demo 会议回放 | 后端 → 前端 |
| `/ws/live` | 实时录音转写 | 双向 |
| `/ws/logs` | 调试日志流 | 后端 → 前端 |

### 8.2 上传处理消息格式

**后端 → 前端：**

```json
// 连接确认
{"type": "connected", "data": {"file_id": "...", "message": "..."}}

// 处理状态
{"type": "status", "data": {"status": "processing", "filename": "...", "engine": "..."}}

// 进度更新
{"type": "progress", "data": {"stage": "vad", "fraction": 0.15}}

// 识别片段 (核心消息)
{
  "type": "segment",
  "data": {
    "segment": { /* ASRResult 完整结构 */ },
    "segment_index": 0,
    "total_estimated": 146,
    "cumulative_stats": {"segments_processed": 1}
  }
}

// 处理完成
{
  "type": "complete",
  "data": {
    "file_id": "...",
    "filename": "...",
    "status": "completed",
    "segments_count": 146,
    "segments": [ /* 所有 segment 的完整列表 */ ]
  }
}

// 错误
{"type": "error", "data": {"message": "..."}}

// 超时
{"type": "timeout", "data": {"message": "Processing timeout (10 min)"}}
```

### 8.3 前端回退策略

```
WebSocket 优先 → 失败则 HTTP 回退 → 30秒超时自动回退
```

---

## 9. 前端架构

### 9.1 组件树

```
index.html (1696行, 单文件 SPA)
├── AudioPlaybackManager (class)
│   ├── create(segIndex, audioBase64, containerEl)  → WaveSurfer实例
│   ├── play(segIndex) / pause() / reset()
│   ├── _b64toBlob()       ← Base64 → Blob 解码
│   ├── _evict()           ← LRU 淘汰 (max 40)
│   └── _updateDBMeter()   ← 伪 dB 表模拟
│
├── 上传管线
│   ├── handleFileUpload()         ← 主入口
│   ├── handleUploadWSMessage()    ← WS 消息分发
│   ├── handleUploadHTTPResponse() ← HTTP 回退渲染
│   ├── uploadFallback()           ← WS 彻底失败时的最终回退
│   └── finishUploadUI()           ← 清理 + 状态恢复
│
├── Demo 模式
│   ├── startDemo()                ← 内嵌演示数据回放
│   └── playNextDemoSegment()      ← 定时器驱动
│
├── 实时录音
│   ├── toggleMic()                ← 开始/停止录音
│   └── ScriptProcessorNode        ← 4 chunks/次批量 WS 发送
│
└── UI 组件
    ├── renderSegment()            ← 转录卡片 + 波形容器
    ├── updateStats()              ← 侧边栏统计
    ├── updateLogicPanel()         ← 逻辑标记面板
    └── checkBackendStatus()       ← 后端心跳检测
```

### 9.2 浏览器兼容性

| 特性 | 要求 |
|------|------|
| ES Module | Chrome 61+, Edge 16+, Firefox 60+ |
| WebSocket | 所有现代浏览器 |
| AudioContext | Chrome 35+, 需 HTTPS 或 localhost |
| wavesurfer.js | CDN 加载 (unpkg.com, 5s 超时) |

---

## 10. 部署指南

### 10.1 环境准备

```bash
# 1. Python 环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
cd backend
pip install -r requirements.txt

# 3. 验证 sherpa-onnx
python -c "import sherpa_onnx; print(sherpa_onnx.__version__)"
```

### 10.2 下载模型

```bash
# ASR 模型 (三个文件, ~300MB)
# 从 k2-fsa/sherpa-onnx Releases 下载并放入:
backend/xasr/models/
  ├── encoder-160ms.onnx
  ├── decoder-160ms.onnx
  ├── joiner-160ms.onnx
  └── tokens.txt

# 标点恢复模型 (~281MB)
mkdir -p backend/xasr/models/punct
cd backend
python -m modules.punctuation_model --download
# 自动从 GitHub 下载并解压到 models/punct/
```

### 10.3 启动

```bash
# 一键启动 (前后端)
cd dickrun
python start.py

# 或分别启动
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8765

cd frontend
python -m http.server 3000
```

### 10.4 无模型运行 (Demo 模式)

如果模型未下载，系统自动进入 Demo 模式：
- 使用预设的模拟会议数据（2个片段）
- 所有 API 正常响应
- Upload 接口返回 demo 数据而非实时转写

---

## 11. 开发指南

### 11.1 模块间依赖关系

```
main.py
  ├── xasr.asr_engine.XASREngine
  │     ├── xasr.sherpa_streaming_infer.SherpaStreamingASR
  │     │     └── sherpa_onnx.OnlineRecognizer
  │     ├── modules.audio_processor
  │     │     ├── estimate_snr()
  │     │     ├── estimate_rt60()
  │     │     ├── HotwordCorrector
  │     │     ├── LogicValidator
  │     │     └── UncertaintyEstimator
  │     └── modules.text_post_processor.process_asr_text()
  │           └── modules.punctuation_model.PunctuationRestorer
  │                 └── sherpa_onnx.OfflinePunctuation
  └── utils.logger
```

### 11.2 添加新认知模块

1. 在 `backend/modules/` 下创建新 `.py` 文件
2. 在 `__init__.py` 中导出
3. 在 `asr_engine.py` 的 `_build_result()` 中调用
4. 通过 `XASREngine.__init__()` 的 `enable_xxx` 参数控制开关

### 11.3 调整 VAD 敏感度

针对不同录音环境，修改 `backend/xasr/asr_engine.py` 中的调用参数：

```python
# 安静的录音室 → 更敏感
segments = _energy_vad(data, sr,
    energy_threshold_ratio=0.03,
    min_silence_frames=20,
)

# 嘈杂的会议室 → 更严格
segments = _energy_vad(data, sr,
    energy_threshold_ratio=0.10,
    min_silence_frames=40,
)
```

### 11.4 切换解码策略

```python
# 默认: 贪心搜索 (最快)
engine = XASREngine(decoding_method="greedy_search")

# 如果 sherpa-onnx 支持: 束搜索 (更准确但更慢)
engine = XASREngine(decoding_method="modified_beam_search")
```

### 11.5 前端调试

打开浏览器控制台 (F12)，查找以下日志标签：

| 标签 | 含义 |
|------|------|
| `[DiTing]` | 通用日志 |
| `[Audio]` | 音频播放 |
| `[Upload]` | 上传流程 |

前端自带调试面板：点击底部 **Log** 按钮可查看后端实时日志。

---

## 12. 已知限制与改进方向

### 12.1 当前限制

| 限制 | 影响 | 临时方案 |
|------|------|----------|
| ASR 模型不含标点 | 需后处理恢复 | CT-Transformer (已集成) |
| 无说话人分离 | 多人会议混为一段 | 规划中 (Embedding + AHC) |
| WAV 编码为 int16 | 量化噪声 -96dB | 可接受 |
| 前端为单文件 | >2000行难以维护 | 可拆分为模块 |
| ffmpeg 未安装 | pydub 使用 librosa 回退 | 安装 ffmpeg 可提速 |
| 端点检测不可靠 | 中文口语特点 | VAD 为主要切句手段 |

### 12.2 改进路线图

| 优先级 | 功能 | 预估工时 |
|--------|------|----------|
| P0 | 说话人分离 (Embedding + AHC 聚类) | 5h |
| P1 | 说话人身份识别 (Eval_Ali 注册库) | 3h |
| P1 | 安装 ffmpeg、优化 pydub 性能 | 0.5h |
| P2 | 替换带标点 ASR 模型 (原生标点输出) | 2h |
| P2 | 前端拆分为 ES 模块 | 4h |
| P3 | GPU 推理支持 (CUDA ONNX) | 3h |
| P3 | 重叠语音检测 (pyannote) | 4h |

---

## 附录 A: 快速参考卡片

```bash
# 启动
cd D:\dickrun && python start.py

# 测试后端
curl http://localhost:8765/api/health

# 命令行处理音频
cd backend && python run_demo.py meeting.mp3

# 下载标点模型
cd backend && python -m modules.punctuation_model --download

# 测试标点模型
cd backend && python -m modules.punctuation_model --test

# 查看日志
tail -f backend/logs/diting.log
```

## 附录 B: Git 仓库

- **仓库地址**: https://github.com/ding060426/dickrun
- **已排除**: 模型文件 (`.onnx`)、音频文件 (`.mp3/.wav`)、数据集 (`Eval_Ali/`)、日志 (`logs/`)
- **提交**: 模型文件需按本文档说明单独下载

---

> **文档版本**: v3.0 | **更新日期**: 2026-07-15 | **作者**: DiTing Team
