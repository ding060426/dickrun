# DiTing 4.5 最终交接说明（diting-finalhugedick 分支）

更新时间：2026-07-15

## 1. 分支与仓库

- 仓库：`https://github.com/ding060426/dickrun`
- 分支：`diting-finalhugedick`
- 主程序目录：`D:\diting4.5`
- 启动入口：`start.py`

本分支包含 DiTing 4.5 当前主体程序、前端页面、后端 FastAPI 服务、本地记忆数据库代码、会议摘要 API 调用配置能力，以及 X-ASR / 标点恢复相关模型文件。

## 2. 启动方式

在 Windows 终端中执行：

```bat
cd /d D:\diting4.5
python start.py
```

启动后访问：

- 前端页面：`http://localhost:3000`
- 后端 API：`http://localhost:8765`
- API 文档：`http://localhost:8765/docs`
- 健康检查：`http://localhost:8765/api/health`
- 历史记忆：`http://localhost:8765/api/memory/history`

`start.py` 当前会在启动前尝试清理旧的 `8765` 和 `3000` 端口进程，避免旧后端占用端口导致新版接口 404。

## 3. Python 环境与依赖

建议使用 Python 3.10+。进入项目后安装后端依赖：

```bat
cd /d D:\diting4.5\backend
pip install -r requirements.txt
```

如果要处理视频或部分音频容器，建议额外安装并配置 `ffmpeg`，否则 `pydub` 可能提示找不到 `ffmpeg/ffprobe`。当前 MP3 在多数情况下可以通过 `librosa` 兜底读取。

## 4. 模型文件说明

本分支通过 Git LFS 管理大模型文件，主要包括：

- `backend/xasr/models/encoder-160ms.onnx`
- `backend/xasr/models/decoder-160ms.onnx`
- `backend/xasr/models/joiner-160ms.onnx`
- `backend/xasr/models/tokens.txt`
- `backend/xasr/models/punct/model.onnx`
- `backend/xasr/models/punct/tokens.json`
- `backend/xasr/models/punct/config.yaml`
- `backend/xasr/models/punct/*.py`
- `backend/xasr/models/punct/README.md`

首次克隆后如果模型文件显示为 LFS 指针，请执行：

```bat
git lfs install
git lfs pull
```

## 5. 核心功能

### 5.1 上传音频/视频识别

前端支持选择：

```html
audio/*,video/*
```

后端仍走现有音频解码与 X-ASR 识别链路。视频是否能解码取决于本机 `ffmpeg/pydub/librosa` 环境。

### 5.2 本地历史记忆

上传识别完成后，后端会把文本与结构化结果保存到本地 SQLite：

- 默认数据库：`backend/storage/diting.db`
- 可通过环境变量覆盖：`DITING_DB_PATH`

保存内容包括：

- 文件名
- ASR 引擎信息
- 分段文本 `segments`
- 完整文本 `full_text`
- 关键术语 `hotwords`
- 会议摘要 `summary`
- 说话人分布 `speaker_stats`
- ASR 优化报告 `asr_optimizer`

不会保存：

- `audio_wav_base64`
- 原始音频/视频二进制内容

历史 API：

```http
GET /api/memory/history?limit=20&offset=0&q=关键词
GET /api/memory/search?q=关键词&limit=20&offset=0
GET /api/memory/history/{meeting_id}
DELETE /api/memory/history/{meeting_id}
```

前端点击“历史”按钮后，会直接显示历史文本预览；点击某条历史记录可把完整历史内容回填到当前主界面。

### 5.3 会议摘要 API 生成

会议摘要已调整为优先调用 LLM API 生成，而不是只依赖本地规则拼接。当前支持 OpenAI 兼容接口，默认配置为 apifusion GPT-5.5。

本地新建 `.env` 文件配置：

```env
DITING_LLM_PROVIDER=gpt55
DITING_LLM_BASE_URL=https://apifusion.aispeech.com.cn
DITING_LLM_MODEL=gpt-5.5
DITING_LLM_API_KEY=你的 API Key
```

注意：`.env` 已被 `.gitignore` 忽略，不应提交真实密钥。

验证方式：

```http
GET http://localhost:8765/api/cognitive/status
```

正常时应看到：

```json
{
  "llm_available": true,
  "llm_provider": "GPT55Client"
}
```

上传文件完成后，后端日志中应出现类似：

```text
POST https://apifusion.aispeech.com.cn/v1/chat/completions 200 OK
Meeting summary generated via LLM API
```

## 6. 前端变化

前端主文件：`frontend/index.html`

主要变化：

1. 删除“会议领域”展示卡片；
2. 保留关键术语、说话人分布、会议摘要；
3. 新增“历史”按钮和历史面板；
4. 历史面板打开后直接显示历史文本预览；
5. 支持历史搜索；
6. 点击历史记录后可回填转写片段、摘要、热词、说话人分布。

## 7. 后端变化

主要文件：

- `backend/main.py`
- `backend/utils/local_db.py`
- `backend/modules/llm_client.py`
- `backend/modules/cognitive_engine.py`
- `backend/modules/asr_optimizer.py`
- `backend/xasr/asr_engine.py`

重点说明：

- `local_db.py` 使用 Python 标准库 `sqlite3`，未引入 ORM；
- 数据库使用 WAL 与独立连接，适配 FastAPI 后台线程；
- 上传完成后通过 `_persist_meeting_safely()` 容错保存，数据库失败不会阻断 ASR 返回；
- `_build_meeting_summary()` 已改为优先走 LLM API，API 不可用时才降级规则摘要；
- `start.py` 会加载 `.env` 中的本地 API 配置。

## 8. 常见问题

### 8.1 历史记忆显示 Not Found

原因通常是旧后端仍占用 `8765`。解决：

```bat
netstat -ano | findstr :8765
taskkill /PID 进程号 /F
cd /d D:\diting4.5
python start.py
```

新版 `start.py` 已尽量自动处理该问题。

### 8.2 会议摘要不是 API 生成

检查：

1. `.env` 是否存在；
2. `DITING_LLM_API_KEY` 是否正确；
3. `/api/cognitive/status` 中 `llm_available` 是否为 `true`；
4. 后端日志是否出现 apifusion 请求记录。

### 8.3 模型缺失

检查：

```bat
git lfs install
git lfs pull
```

并确认：

```text
backend/xasr/models/encoder-160ms.onnx
backend/xasr/models/decoder-160ms.onnx
backend/xasr/models/joiner-160ms.onnx
backend/xasr/models/tokens.txt
backend/xasr/models/punct/model.onnx
```

## 9. 协作注意事项

1. 不要提交 `.env`、数据库、音视频文件、日志文件；
2. 数据库 `backend/storage/diting.db` 是本地运行数据，不应进入 Git；
3. 如果修改历史记忆结构，应同步更新 `backend/utils/local_db.py` 的兼容字段迁移逻辑；
4. 如果修改上传返回字段，应保持现有字段兼容，避免影响队友模块；
5. 实时麦克风 `/ws/live` 当前不写入本地记忆，本地记忆只保存上传音视频的识别结果。
