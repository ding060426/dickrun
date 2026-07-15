# 谛听 (DiTing) v3.0 — Smart Meeting Speech Cognitive System

基于 X-ASR (sherpa-onnx zipformer2) 的智能会议语音认知系统。支持音频文件上传、实时转写、热词修正、逻辑校验、音频波形可视化。

## 系统架构

```
frontend/
  index.html       ← 前端 SPA (wavesurfer.js 波形 + WebSocket)
backend/
  main.py          ← FastAPI 后端
  modules/
    supabase_db.py ← Supabase 数据库操作 (默认)
    meeting_db.py  ← SQLite 数据库操作 (备选)
  xasr/
    asr_engine.py  ← X-ASR 引擎 (VAD + 端点检测 + 转写)
    models/        ← ONNX 模型 (需自行下载)
  utils/
    logger.py      ← 日志系统 (控制台 + 文件 + 环形缓冲)
  supabase_init.sql ← Supabase 数据库建表 SQL
```

## 数据库

### Supabase (默认)

项目默认连接 Supabase 云数据库。在 `backend/.env` 中配置：

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=sb_publishable_xxx
```

#### 初始化数据库

1. 登录 [Supabase Dashboard](https://supabase.com/dashboard)
2. 进入项目 → **SQL Editor**
3. 复制 `backend/supabase_init.sql` 全部内容，点击 **Run** 执行

`supabase_init.sql` 包含以下表：

| 表名 | 说明 |
|------|------|
| `users` | 用户表 |
| `auth_sessions` | 认证会话表 |
| `meeting_reservations` | 会议预约表 |
| `meeting_joins` | 参会记录表 |
| `meeting_analyses` | 会议分析表 |
| `friends` | 好友/同事关系表 |

#### 管理员账号

执行 `backend/setup_supabase.py` 创建默认管理员：
```
python backend/setup_supabase.py
```
默认账号: `admin / admin123`

### SQLite (备选)

如需使用本地 SQLite，修改 `backend/main.py` 第 39 行：

```python
# from modules import supabase_db as db    # Supabase
from modules import meeting_db as db        # SQLite
```

SQLite 数据文件存储在 `backend/data/diting.db`，首次启动自动建表。

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn numpy supabase
```

### 2. 下载 X-ASR 模型文件

模型文件约 310MB：

```bash
mkdir -p backend/xasr/models
cd backend/xasr/models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-06-28.tar.bz2
tar -xjf sherpa-onnx-zipformer-zh-en-2023-06-28.tar.bz2
```

模型目录应有：
```
backend/xasr/models/
  ├── encoder-160ms.onnx    (~295 MB)
  ├── decoder-160ms.onnx    (~2.5 MB)
  ├── joiner-160ms.onnx     (~2.0 MB)
  └── tokens.txt            (~20 KB)
```

### 3. 启动

```bash
python start.py
```

- 前端: http://localhost:3000
- 后端 API: http://localhost:8765
- API 文档: http://localhost:8765/docs

模型未就绪时自动进入 Demo 模式。

## 功能

| 功能 | 状态 | 说明 |
|------|------|------|
| ASR 转写 | ✅ | sherpa-onnx zipformer2 流式推理 |
| VAD + 端点检测 | ✅ | 能量检测 + 动态阈值 + 片段合并 |
| 热词修正 | ✅ | 拼音匹配 + 模糊音校正 |
| 逻辑校验 | ✅ | 数据冲突检测 |
| 音频波形可视化 | ✅ | wavesurfer.js + WAV 编码 |
| 实时录音转写 | ✅ | WebSocket 流式传输 |
| 会议预约 | ✅ | 日历视图 + 参会码 |
| 好友/同事管理 | ✅ | 搜索添加 + 同事列表 |
| 会议分析 | ✅ | 转写记录保存 |
| Eval_Ali 评测 | ✅ | CER 计算 / 热词提取 |

## License

MIT
