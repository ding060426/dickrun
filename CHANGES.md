# DiTing 谛听 - 改动升级记录

> 本文档记录截至 2026-07-15 (v4.5) 的所有架构变更、功能新增、Bug 修复和工程优化，用于 GitHub 版本管理参考。

---

## 目录

- [1. ASR 核心修复](#1-asr-核心修复)
- [2. 文本后处理管线](#2-文本后处理管线)
- [3. VAD 语音分段系统](#3-vad-语音分段系统)
- [4. 置信度计算系统](#4-置信度计算系统)
- [5. 行动项自动提取](#5-行动项自动提取)
- [6. 前端升级](#6-前端升级)
- [7. Bug 修复与工程清理](#7-bug-修复与工程清理)
- [8. 文件变更清单](#8-文件变更清单)
- [9. 架构总览](#9-架构总览)
- [10. VAD 真实音频验证体系](#10-vad-真实音频验证体系)
- [11. 上传任务状态持久化/任务管理](#11-上传任务状态持久化任务管理)
- [12. 前端拆分](#12-前端拆分)
- [13. 版本号统一更新](#13-版本号统一更新)

---

## 1. ASR 核心修复

### 1.1 Token 词表修复

**问题**：ASR 输出乱码（如 `<0x0D>p-�窗猪01`），识别准确率为 0%。

**根因**：`backend/xasr/models/tokens.txt` 仅含 2002 个 token，实际需要 5000 BPE tokens。

**修复**：
- 用 `backend/xasr/zipformer/data/lang_5000_with_punctuation/tokens.txt`（5000 tokens）替换错误文件
- 旧文件备份为 `tokens.txt.wrong_backup`
- 在 `download_models.py` 中添加 token 数量校验

**效果**：识别准确率从 0% 提升至 84.3%（CER 15.7%），在 test_data 和 Eval_Ali 数据集上验证通过。

### 1.2 sherpa-onnx Token 信息暴露

**文件**：`backend/xasr/sherpa_streaming_infer.py`

新增 `get_token_info()` 方法，暴露 sherpa-onnx 底层的 token 级概率信息：

```python
def get_token_info(self) -> dict:
    tokens = list(self.recognizer.tokens(self.stream))
    timestamps = list(self.recognizer.timestamps(self.stream))
    ys_probs = list(self.recognizer.ys_probs(self.stream))
    return {"tokens": tokens, "timestamps": timestamps, "ys_probs": ys_probs}
```

这为后续的 token 级置信度计算提供了数据基础。

---

## 2. 文本后处理管线

### 2.1 统一文件

将两个重复文件合并为统一管线：
- **删除** `backend/modules/text_postprocessor.py`（旧文件，仅规则+MacBERT）
- **保留** `backend/modules/text_post_processor.py`（统一版，432 行）
- `main.py` 导入简化为 `from modules.text_post_processor import postprocess_text`

### 2.2 六步处理管线

```
raw ASR text → ① 语气词过滤 → ② 重复词合并 → ③ 标点恢复(模型优先, 规则兜底)
              → ④ 强制断句 → ⑤ MacBERT纠错(可选) → ⑥ 规范化
```

| 步骤 | 功能 | 延迟 |
|------|------|------|
| ① 语气词过滤 | 移除句首/句尾语气词（嗯、额、啊等）和句中口头禅 | <1ms |
| ② 重复词合并 | `是是是`→`是`，`好的好的`→`好的` | <1ms |
| ③ 标点恢复 | sherpa-onnx CT-Transformer 模型优先，规则兜底 | 10-50ms |
| ④ 强制断句 | 超过40字无标点时在弱语义边界插入逗号 | <1ms |
| ⑤ MacBERT纠错 | pycorrector 同音字/形似字纠错（真→正，他→她） | 50-200ms |
| ⑥ 规范化 | 重复字合并、中英文间距、标点清理 | <1ms |

### 2.3 双接口设计

- `process_asr_text(text) -> str`：返回纯文本，用于流式处理
- `postprocess_text(text) -> (str, dict)`：返回文本+详细信息（含纠错列表、语气词列表），用于最终展示

### 2.4 标点恢复模型

`backend/modules/punctuation_model.py`（213 行）：
- 优先使用 sherpa-onnx `OfflinePunctuation`（CT-Transformer 模型）
- 模型不可用时自动降级到规则引擎（断句关键词 + 疑问/感叹结尾检测）

### 2.5 MacBERT 纠错

- 使用 `pycorrector.MacBertCorrector`（完整版，非自研轻量版）
- 模型懒加载，首次调用约 2 秒，后续即时响应
- torch 不可用时自动跳过，不影响管线其他步骤

---

## 3. VAD 语音分段系统

### 3.1 三级 Fallback 架构

**文件**：`backend/modules/vad_manager.py`（523 行）

```
segment_audio()
  ├─ FireRedVAD (首选, 97.57% F1)
  ├─ Silero VAD (备选, 95.95% F1)
  └─ Energy VAD (兜底)
```

| VAD 引擎 | 模型大小 | F1 | 流式支持 | 特点 |
|----------|---------|-----|---------|------|
| FireRedVAD | 2.2MB (Stream-VAD) | 97.57% | 是 (`detect_full`) | DFSMN 神经网络，精度最高 |
| Silero VAD | 629KB | 95.95% | 是 (`accept_waveform`) | sherpa-onnx 内置 |
| Energy VAD | 无 | ~85% | 是 | 纯能量阈值，最后兜底 |

### 3.2 分段参数

| 参数 | 值 | 作用 |
|------|-----|------|
| `max_duration` | 30s | 防止过长分段 |
| `min_silence_duration` | 500ms | 避免在呼吸停顿内切分 |
| `min_segment_duration` | 0.5s | 过滤噪声碎片 |
| `preroll` | 200ms | 保留句首 |
| `postroll` | 200ms | 保留句尾 |

### 3.3 流式 VAD 封装

`StreamVADState` 类提供实时麦克风流式 VAD：
- `accept_waveform(chunk)` → 返回 `(event_type, timestamp)` 元组
- 支持三种 VAD 后端切换
- 事件类型：`speech_start`、`speech_end`

### 3.4 ASR 引擎集成

`backend/xasr/asr_engine.py` 的 `process_file()` 方法：
- 调用 `vad_manager.segment_audio()` 进行分段
- 对每个分段独立进行 ASR 识别
- 测试结果（120s Eval_Ali 音频）：FireRedVAD 产生 8 段，平均 5.0s，最短 1.6s，无超 30s 分段

### 3.5 API 端点

`GET /api/vad/status` — 返回当前 VAD 系统状态：
```json
{
  "available": true,
  "firered_available": true,
  "silero_available": true,
  "active_vad": "firered",
  "priority": "FireRedVAD → Silero → Energy"
}
```

---

## 4. 置信度计算系统

### 4.1 问题根因

原置信度仅依赖 SNR/RT60 声学指标（`max(0.4, snr_score * 0.9)`），导致无论实际语音清晰度如何，置信度始终在 0.4-0.6 之间。

### 4.2 三层置信度公式

```
final_confidence = 0.6 * token_confidence + 0.3 * acoustic_quality + 0.1 * text_bonus
```

| 层级 | 权重 | 数据来源 | 说明 |
|------|------|---------|------|
| Token 概率 | 60% | sherpa-onnx `ys_probs` | token 级对数概率的指数化均值 |
| 声学质量 | 30% | SNR + RT60 | 信噪比 + 混响时间 |
| 文本加成 | 10% | 后处理纠错数 | 纠错少 = 高置信度 |

### 4.3 修改文件

| 文件 | 改动 |
|------|------|
| `sherpa_streaming_infer.py` | 新增 `get_token_info()` 暴露 `ys_probs` |
| `audio_processor.py` | 新增 `compute_token_confidence()` 函数 |
| `asr_engine.py` | `_build_result()` 方法重写，使用三层公式 |
| `frontend/index.html` | 置信度彩色显示（绿>0.8, 黄0.5-0.8, 红<0.5）+ 不确定词高亮 |

### 4.4 效果

- 清晰语音：置信度 90%+（绿色显示）
- 低质量分段：置信度 <50%（红色显示），不确定词高亮标记
- 置信度范围从 0.4-0.6 扩展到 0.1-0.99

---

## 5. 行动项自动提取

### 5.1 功能概述

从会议转写文本中自动提取：
- **谁负责**（assignee）
- **做什么**（task）
- **截止时间**（deadline）
- **优先级**（priority: high/medium/low）
- **相关原文片段**（source_text）

### 5.2 双轨架构

**文件**：`backend/modules/action_extractor.py`（334 行）

```
ActionExtractor.extract(segments)
  ├─ _extract_with_rules()   ← 规则模式匹配 (优先, <10ms)
  └─ _extract_with_llm()     ← LLM 语义理解 (可选, 1-3s)
```

### 5.3 规则模式

| 模式类型 | 正则示例 | 匹配示例 |
|---------|---------|---------|
| 负责人 | `由?(.{2,4})负责` | "张三负责整理报告" |
| 截止时间 | `(本周|周五|月底|下周|明天|今天|前)` | "周五前完成" |
| 任务动词 | `整理|完成|优化|提交|确认|跟进` | "优化VAD参数" |
| 优先级关键词 | `紧急|尽快|立即` → high | "紧急处理" |

### 5.4 API 端点

`POST /api/action-items`

```json
// Request
{
  "segments": [
    {"speaker": "张三", "text": "李四负责整理测试报告，周五前完成", "start": 0.0, "end": 5.0}
  ]
}

// Response
{
  "action_items": [
    {
      "assignee": "李四",
      "task": "整理测试报告",
      "deadline": "周五前",
      "priority": "medium",
      "source_text": "李四负责整理测试报告，周五前完成"
    }
  ],
  "method": "rules"
}
```

### 5.5 集成

- `cognitive_engine.py` 中 `_extract_actions()` 调用
- 前端 `stopMicRecording()` 结束录音后自动触发提取
- 前端 `fetchAndRenderActionItems()` 渲染优先级彩色卡片

---

## 6. 前端升级

### 6.1 版本 v3.0 → v3.1

**文件**：`frontend/index.html`（2252 行，单文件 SPA）

### 6.2 新增功能

| 功能 | 说明 |
|------|------|
| 麦克风 Orb 动画 | 录音状态可视化球体动画 |
| 四面板分析结果 | 转写、认知分析、行动项、调试日志分区展示 |
| 双通道文件上传 | WebSocket 上传 + HTTP fallback（带取消功能） |
| 分段波形播放 | wavesurfer.js 分段音频回放 |
| 调试日志面板 | 按严重级别色彩编码（DEBUG/INFO/WARN/ERROR） |
| 置信度彩色显示 | 绿/黄/红三色 + 不确定词高亮 |
| 行动项卡片 | 按优先级色彩渲染 |

### 6.3 核心函数

- `confidenceColor(confidence)` → 返回颜色值
- `confidenceLabel(confidence)` → 返回标签文本
- `fetchAndRenderActionItems()` → 获取并渲染行动项
- `updateMicResult(data)` → 更新麦克风识别结果（含彩色置信度）
- `stopMicRecording()` → 结束录音并自动触发行动项提取

---

## 7. Bug 修复与工程清理

### 7.1 JSONResponse 导入缺失（HIGH）

**问题**：`/api/action-items` 端点使用了 `JSONResponse` 但未导入。
**修复**：`main.py` 第29行统一为 `from fastapi.responses import JSONResponse, HTMLResponse, FileResponse`，移除重复导入行。

### 7.2 Request 未定义（HIGH）

**问题**：`extract_actions` 函数签名使用 `request: Request` 但 `Request` 未导入，导致 `start.py` 启动崩溃。
**修复**：`main.py` 第28行 `from fastapi import` 添加 `Request`。

### 7.3 重复文件合并（MEDIUM）

**问题**：`text_postprocessor.py` 和 `text_post_processor.py` 两个文件功能重叠。
**修复**：
- 合并到统一的 `text_post_processor.py`
- 删除 `text_postprocessor.py`
- 简化 `main.py` 导入逻辑（移除 try/except fallback 链）
- `__init__.py` 导出 `postprocess_text`

### 7.4 numpy float32 序列化（LOW）

**问题**：numpy float32 类型无法直接 JSON 序列化，导致 WebSocket 消息静默失败。
**修复**：在 `_build_result()` 中显式转换为 Python 原生类型（`float()`/`str()`/`bool()`）。

### 7.5 Silero VAD 缓冲区溢出（LOW）

**问题**：`buffer_size_in_seconds=120` 无法处理 1551s 长音频。
**修复**：改为按 60 秒批次处理。

### 7.6 sherpa-onnx API 适配（LOW）

- `accept_waveform` 实际签名是 `accept_waveform(chunk)` 而非 `accept_waveform(sample_rate, chunk)`
- `seg.start` 单位是 samples 而非 seconds，需 `/ sample_rate` 转换
- FireRedVAD `detect_full` 签名是 `detect_full(i16)` 而非 `detect_full(i16, sr)`

---

## 8. 文件变更清单

### 8.1 新增文件

| 文件路径 | 行数 | 说明 |
|---------|------|------|
| `backend/modules/vad_manager.py` | 523 | 三级 VAD fallback 管理器 |
| `backend/modules/action_extractor.py` | 334 | 行动项提取（规则+LLM） |
| `backend/modules/text_post_processor.py` | 432 | 统一文本后处理管线 |
| `backend/modules/punctuation_model.py` | 213 | CT-Transformer 标点恢复 |
| `backend/modules/cognitive_engine.py` | 433 | 认知引擎（内容预测、领域推断、摘要） |
| `backend/modules/domain_taxonomy.py` | 169 | 领域分类法 |
| `backend/modules/hotword_engine.py` | 393 | 热词引擎 |
| `backend/modules/llm_client.py` | 421 | LLM 客户端（GPT/DeepSeek/Qwen/Mock） |
| `backend/modules/speaker_diarization.py` | 673 | 说话人分离 |
| `backend/test_asr_file.py` | 200 | VAD+ASR 验证工具（argparse + JSON） |
| `backend/xasr/models/silero_vad.onnx` | 629KB | Silero VAD 模型 |
| `backend/xasr/models/firered_vad/` | ~2.2MB | FireRedVAD 模型 (3子模型) |
| `frontend/css/styles.css` | 489 | 前端 CSS 外置 |
| `frontend/js/app.js` | 1672 | 前端 JS 外置（ES Module） |

### 8.2 修改文件

| 文件路径 | 行数 | 主要改动 |
|---------|------|---------|
| `backend/main.py` | 1036 | JSONResponse导入、Request导入、VAD/Action/Post-processor集成、上传任务管理、26路由+4 WS |
| `backend/xasr/asr_engine.py` | 1053 | VAD集成、三层置信度 `_build_result()` 重写 |
| `backend/xasr/sherpa_streaming_infer.py` | 174 | 新增 `get_token_info()` |
| `backend/modules/audio_processor.py` | 542 | 新增 `compute_token_confidence()` |
| `frontend/index.html` | 178 | v4.5 前端，CSS/JS 外置后仅保留 HTML 结构 |
| `backend/modules/__init__.py` | 60 | 导出统一 |
| `start.py` | 116 | 版本号更新为 v4.5 |
| `.gitignore` | - | 新增排除规则 |

### 8.3 删除文件

| 文件路径 | 说明 |
|---------|------|
| `backend/modules/text_postprocessor.py` | 重复文件，功能合并到 `text_post_processor.py` |

### 8.4 替换文件

| 文件路径 | 说明 |
|---------|------|
| `backend/xasr/models/tokens.txt` | 2002 tokens → 5000 BPE tokens（旧文件备份为 `.wrong_backup`） |

---

## 9. 架构总览

### 9.1 API 端点（26 HTTP + 4 WebSocket）

| 类别 | 端点 | 方法 |
|------|------|------|
| 系统 | `/api/health`, `/api/xasr/status`, `/api/cognitive/status`, `/api/vad/status` | GET |
| 会议 | `/api/meeting/demo` | GET |
| 热词 | `/api/hotwords` | GET/POST |
| 日志 | `/api/logs/recent`, `/api/logs/download` | GET |
| Eval_Ali | `/api/eval/status`, `/api/eval/hotwords`, `/api/eval/meeting/{id}` | GET |
| 说话人 | `/api/speakers`, `/api/speakers/enroll`, `/api/speakers/enroll_from_eval` | GET/POST |
| 领域 | `/api/domain/taxonomy`, `/api/domain/infer` | GET/POST |
| 行动项 | `/api/action-items` | POST |
| 音频 | `/api/audio/upload`, `/api/audio/upload/{id}/cancel`, `/api/audio/upload/{id}/status` | POST/GET |
| WebSocket | `/ws/meeting`, `/ws/live`, `/ws/upload/{id}`, `/ws/logs` | WS |

### 9.2 模块依赖关系

```
main.py
  ├─ xasr/asr_engine.py
  │    ├─ xasr/sherpa_streaming_infer.py  (get_token_info)
  │    ├─ modules/vad_manager.py          (segment_audio)
  │    └─ modules/audio_processor.py       (compute_token_confidence)
  ├─ modules/text_post_processor.py        (postprocess_text)
  │    └─ modules/punctuation_model.py     (CT-Transformer)
  ├─ modules/action_extractor.py            (extract_action_items)
  ├─ modules/cognitive_engine.py
  │    ├─ modules/domain_taxonomy.py
  │    ├─ modules/hotword_engine.py
  │    └─ modules/llm_client.py
  └─ modules/speaker_diarization.py
```

### 9.3 双轨设计原则

所有认知模块遵循 **LLM/模型优先 → 规则兜底** 的双轨架构：

| 模块 | 模型/LLM 优先 | 规则兜底 |
|------|-------------|---------|
| 标点恢复 | CT-Transformer | 断句关键词 + 结尾检测 |
| MacBERT纠错 | pycorrector | 跳过 |
| 行动项提取 | LLM 语义理解 | 正则模式匹配 |
| 领域推断 | LLM | 关键词匹配 |
| VAD | FireRedVAD/Silero | Energy |

确保在无外部依赖时系统仍可运行。

---

## 待办事项

- [ ] `TECHNICAL_DOCS.md` 需要完整重写（当前内容已严重过时）
- [x] ~~`start.py` 版本号仍为 v2.0，需更新为 v3.1~~ → 已更新为 v4.5
- [ ] `README.md` 特性表缺少行动项、LLM、领域推断等功能
- [ ] 清理 `model_downloads/` 中的不完整下载文件
- [x] ~~前端 HTML 文件结构优化（2252 行单文件）~~ → 已完成第一轮 CSS/JS 拆分

---

## 10. VAD 真实音频验证体系

### 10.1 test_asr_file.py 升级

**文件**：`backend/test_asr_file.py`（200 行，从 41 行升级）

从临时脚本升级为可重复运行的真实音频验证工具，支持三种模式：

```bash
python backend/test_asr_file.py <audio_path>           # VAD + ASR 完整验证
python backend/test_asr_file.py <audio_path> --no-asr   # 仅 VAD 分段
python backend/test_asr_file.py <audio_path> --json     # JSON 格式报告
```

**输出指标**：

| 指标 | 说明 |
|------|------|
| `audio_duration_sec` | 音频总时长 |
| `sample_rate` | 采样率 |
| `vad_type` | 实际使用的 VAD 后端 |
| `vad_segments_count` | VAD 分段数 |
| `speech_coverage_ratio` | 语音覆盖率 |
| `min/max/avg_segment_duration` | 分段时长统计 |
| `asr_segments_count` | ASR 识别段数 |
| `non_empty_text_segments` | 非空文本段数 |
| `total_text_chars` | 文本总字数 |
| `avg_confidence` | 平均置信度 |
| `elapsed_seconds` | 处理耗时 |
| `rtf` | 实时率（Real-Time Factor） |

验证时默认关闭认知增强项（`enable_logic_validation=False`、`enable_hotword_correction=False`、`enable_uncertainty=False`、`enable_cognitive=False`），避免干扰 VAD/ASR 验证。

### 10.2 vad_manager.py 增强

**文件**：`backend/modules/vad_manager.py`（523 行）

低风险补充，不改核心算法：

- `get_vad_info()` 返回增强：FireRed 模型路径、Silero 模型路径、文件存在性、当前 active VAD、priority
- 新增 `summarize_segments(segments, total_duration)` 返回分段统计摘要，供测试脚本和 API 复用

### 10.3 /api/vad/status 增强

返回更完整的 `get_vad_info()` 信息，包括模型路径和可用性。

---

## 11. 上传任务状态持久化/任务管理

### 11.1 问题

原上传任务状态分散在 `upload_sessions`、`upload_cancel_events`、worker 线程局部变量和前端变量中，存在以下问题：
- WebSocket 断开后状态丢失
- 刷新页面无法恢复任务
- reset 只清 UI，不取消后端任务
- 没有任务状态查询 API

### 11.2 后端进程内任务管理

**文件**：`backend/main.py`

新增进程内任务状态表（不引入数据库）：

```python
upload_tasks: dict = {}
upload_tasks_lock = threading.RLock()
UPLOAD_TERMINAL_STATUSES = {"completed", "error", "cancelled", "demo_mode"}
```

**任务结构**：

```json
{
  "file_id": "...",
  "filename": "...",
  "status": "processing",
  "size_mb": 0,
  "engine": "X-ASR (sherpa-onnx zipformer2 v4.5)",
  "created_at": 0,
  "updated_at": 0,
  "progress_stage": "vad",
  "progress_fraction": 0.0,
  "segments_count": 0,
  "segments": [],
  "summary": null,
  "domain": null,
  "hotwords": [],
  "speaker_stats": {},
  "error": null,
  "cancel_requested": false
}
```

**Helper 函数**：

| 函数 | 说明 |
|------|------|
| `_now_ts()` | 当前时间戳 |
| `_new_upload_task(file_id, ...)` | 创建新任务 |
| `_update_upload_task(file_id, **fields)` | 更新任务字段 |
| `_append_upload_segment(file_id, ...)` | 追加识别段 |
| `_get_upload_task(file_id)` | 获取任务快照 |
| `_serialize_upload_task(task, include_segments)` | 序列化任务 |
| `_cleanup_upload_tasks()` | 清理已终态任务（TTL） |

### 11.3 多 Queue 广播

`upload_sessions` 从单 queue 改为多 queue（`set`），支持同一任务重连或多页面订阅：

```python
upload_sessions[file_id] = set([queue])
```

### 11.4 新增 status API

```
GET /api/audio/upload/{file_id}/status
```

返回任务 snapshot。若不存在返回 `{"ok": false, "error": "not_found"}`。

### 11.5 取消语义修正

`POST /api/audio/upload/{file_id}/cancel`：
- 任务已终态 → 返回当前状态
- 任务运行中 → 设置 `cancel_requested=True`、更新状态为 `cancelling`
- worker 实际退出后更新为 `cancelled`

### 11.6 WebSocket snapshot 恢复

`/ws/upload/{file_id}` 连接后：
- 绑定 queue 到 `upload_sessions[file_id]`
- 发送 `connected` 消息
- 如果任务存在，发送 `{"type": "snapshot", "data": {"task": ...}}`
- 断开时只移除当前 queue，不删除 task

### 11.7 前端任务恢复

**文件**：`frontend/js/app.js`

| 函数 | 说明 |
|------|------|
| `saveUploadTaskRef(fileId, filename)` | 保存到 localStorage |
| `clearUploadTaskRef()` | 清除 localStorage |
| `fetchUploadStatus(fileId)` | 调用 status API |
| `restoreUploadTaskOnLoad()` | 页面加载时恢复任务 |
| `renderUploadSnapshot(task)` | 渲染任务快照 |

使用 `localStorage` 键 `diting:lastUploadTask` 保存最近任务引用。`initApp()` 启动时自动调用 `restoreUploadTaskOnLoad()`。

---

## 12. 前端拆分

### 12.1 第一轮拆分（已完成）

将 `frontend/index.html` 从 2252 行单文件拆分为三部分：

| 文件 | 行数 | 说明 |
|------|------|------|
| `frontend/index.html` | 178 | 纯 HTML 结构 + CSS/JS 引用 |
| `frontend/css/styles.css` | 489 | 全部 CSS 样式 |
| `frontend/js/app.js` | 1672 | 全部 JS 逻辑（ES Module） |

`index.html` 使用外置引用：

```html
<link rel="stylesheet" href="./css/styles.css">
<script type="module" src="./js/app.js"></script>
```

### 12.2 设计原则

- 不引入 npm/Vite/Webpack 等构建工具
- 使用浏览器原生 ES Module
- 保持 `python start.py` 和 Python 静态服务可运行
- 第一轮只搬迁 CSS/JS，不改业务逻辑
- 保留 segment 播放按钮和 upload complete 结果面板触发逻辑

### 12.3 后续拆分计划

第一轮验证通过后，可按模块继续拆分 JS：
`config.js` → `utils.js` → `audio.js` → `upload.js` → `logs.js` → `liveMic.js` → `results.js`

---

## 13. 版本号统一更新

### 13.1 版本升级 v2.0/v3.0 → v4.5

| 文件 | 更新位置 | 旧版本 | 新版本 |
|------|---------|--------|--------|
| `backend/main.py` | docstring | v2.0 | v4.5 |
| `backend/main.py` | FastAPI version | 2.0.0 | 4.5.0 |
| `backend/main.py` | /api/health service | DiTing v2.0 | DiTing v4.5 |
| `backend/main.py` | engine 字符串 (6处) | v2.0 | v4.5 |
| `backend/main.py` | 启动打印 | DiTing v2.0 | DiTing v4.5 |
| `start.py` | docstring | v2.0 | v4.5 |
| `start.py` | 启动日志 | v2.0 | v4.5 |
| `start.py` | 启动打印 | DiTing v2.0 | DiTing v4.5 |
| `frontend/index.html` | title | v3.0 | v4.5 |
| `frontend/index.html` | header | DiTing v3.0 | DiTing v4.5 |
| `frontend/js/app.js` | 注释 | v3.0 | v4.5 |
| `frontend/js/app.js` | 注释 | v3.1 | v4.5 |

### 13.2 验证

- Python `py_compile` 全部通过
- `node --check frontend/js/app.js` 通过
- 所有版本引用已统一为 v4.5
