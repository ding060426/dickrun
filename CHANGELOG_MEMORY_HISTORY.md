# DiTing 4.5 本地记忆功能更新日志

更新时间：2026-07-15

## 1. 更新目标

本次更新面向 DiTing 4.5 主程序，目标是把“上传音视频后的识别文本”沉淀为本地可查询的会议记忆，便于后续更稳定地生成关键术语、会议摘要、说话人分布和跨会议总结。

同时，前端删去了“会议领域”展示区，将界面重点收敛到：

- 关键术语
- 说话人分布
- 会议摘要
- 历史记忆

## 2. 主要改动

### 2.1 新增本地 SQLite 记忆数据库

新增文件：

- `backend/utils/local_db.py`

默认数据库路径：

- `backend/storage/diting.db`

可通过环境变量覆盖：

```bash
DITING_DB_PATH=D:\your_path\diting.db
```

数据库使用 Python 标准库 `sqlite3`，没有新增第三方依赖。

### 2.2 保存上传识别结果

修改文件：

- `backend/main.py`

现在上传音频/视频完成后，后端会自动保存：

- 文件名
- ASR 引擎信息
- 全部转写文本 `full_text`
- 分段 `segments`
- 关键术语 `hotwords`
- 会议摘要 `summary`
- 说话人分布 `speaker_stats`
- ASR 优化报告 `asr_optimizer`

注意：数据库不会保存 `audio_wav_base64`，避免本地数据库体积过大。

### 2.3 新增记忆 API

新增接口：

```http
GET /api/memory/history?limit=20&offset=0&q=关键词
GET /api/memory/search?q=关键词&limit=20&offset=0
GET /api/memory/history/{meeting_id}
DELETE /api/memory/history/{meeting_id}
```

用途：

- `/api/memory/history`：查看历史记录列表；
- `/api/memory/search`：搜索历史全文、文件名、摘要、热词；
- `/api/memory/history/{meeting_id}`：查看完整历史详情；
- `DELETE`：删除单条本地历史记录。

### 2.4 删除“会议领域”前端展示

修改文件：

- `frontend/index.html`

删除内容：

- “会议领域”卡片；
- `domainCard` 引用；
- `renderDomainCard()` 函数；
- 会议领域相关 CSS；
- `renderResultsPanel()` 中领域渲染分支。

后端仍可保留领域推断能力供摘要或后续逻辑使用，只是不再在当前主界面展示。

### 2.5 新增“历史”面板

修改文件：

- `frontend/index.html`

新增功能：

- 控制栏新增“历史”按钮；
- 左下角新增历史记忆面板；
- 支持搜索历史会议；
- 点击历史记录后，可将历史转写片段、关键术语、说话人分布、会议摘要回填到当前界面；
- 快捷键：`Ctrl + H` 打开/关闭历史面板。

### 2.6 支持选择视频文件

前端上传 input 已从：

```html
accept="audio/*"
```

调整为：

```html
accept="audio/*,video/*"
```

后端仍走现有音频解码链路。视频文件能否解析取决于本地 `pydub/ffmpeg` 环境是否可用。

## 3. 不变内容

本次更新不改动：

- `start.py` 启动方式；
- `/ws/live` 实时语音接口；
- `/ws/upload/{file_id}` 上传进度接口协议；
- `XASREngine.process_file()` 主 ASR 流程；
- 已有关键术语、会议摘要、说话人分布的返回字段。

上传完成返回中只新增：

```json
"memory_id": "当前 file_id"
```

不会破坏现有前端和队友模块对接。

## 4. 本地数据库表

### meetings

保存会议级信息：

- `id`
- `filename`
- `status`
- `engine`
- `segments_count`
- `duration_sec`
- `full_text`
- `summary_json`
- `hotwords_json`
- `speaker_stats_json`
- `asr_optimizer_json`
- `metadata_json`
- `created_at`
- `updated_at`

### segments

保存分段级信息：

- `meeting_id`
- `segment_index`
- `speaker_id`
- `start_sec`
- `end_sec`
- `text`
- `raw_text`
- `asr_confidence`
- `snr_db`
- `quality_score`
- `quality_label`
- `terms_json`
- `data_points_json`
- `corrections_json`
- `logic_flags_json`
- `uncertain_spans_json`
- `uncertainty_json`

## 5. 启动与验证

启动方式不变：

```bash
cd D:\diting4.5
python start.py
```

后端：

```text
http://localhost:8765
```

前端：

```text
http://localhost:3000
```

建议验证：

1. 打开前端，确认不再显示“会议领域”；
2. 上传音频或视频；
3. 上传完成后确认关键术语、说话人分布、会议摘要正常显示；
4. 点击“历史”；
5. 确认刚上传的文件出现在历史列表；
6. 点击历史记录，确认转写片段和分析结果可以回填；
7. 重启程序后再次打开历史，确认记录仍存在。

## 6. 已知限制

1. 视频上传依赖本机 `ffmpeg/pydub` 解码能力。如果视频无法解析，请先转成 `.wav` 或 `.mp3`。
2. 当前历史搜索使用 SQLite `LIKE` 查询，没有启用 FTS5 全文索引，优先保证兼容性。
3. 实时麦克风识别暂不写入本地记忆；本次只保存上传音视频处理结果。
4. 数据库文件属于本地运行数据，已在 `.gitignore` 中忽略，不建议提交到 GitHub。

## 7. 给组员的对接说明

- 若需要读取历史会议文本，请调用：

```http
GET /api/memory/history/{meeting_id}
```

重点字段：

- `full_text`：完整会议文本；
- `segments`：逐段文本与说话人；
- `hotwords`：关键术语；
- `summary`：会议摘要；
- `speaker_stats`：说话人分布。

- 若需要跨会议检索，请调用：

```http
GET /api/memory/search?q=关键词
```

- 若后续要做“跨会议长期总结”，建议直接基于 `full_text + summary + hotwords + speaker_stats` 聚合，而不要重新读取音频。
