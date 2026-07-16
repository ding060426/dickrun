# 本地会议记录系统

## 与 `diting-finalhugedick` 的关系

远端分支的历史记忆由 `backend/utils/local_db.py`、`/api/memory/*` 和前端历史抽屉组成。可复用的部分是 SQLite 的会议/分段两层模型、全文字段和历史搜索思路。

该实现不会保存分段音频，也不处理实时麦克风记录，因此本分支没有直接复制它，而是接入当前仓库已有的上传分段音频、实时录音和最终说话人分离管线。

## 本地数据

默认数据库为 `backend/data/records.db`，可通过 `DITING_RECORDS_DB_PATH` 覆盖。

- `meeting_records`：标题、来源、原文件信息、说话人列表、完整纯文本、时长和所有者。
- `meeting_record_segments`：分段时间、说话人、提取文字、识别元数据和 WAV 音频 BLOB。

列表和搜索只读取会议级信息，不会加载大体积分段音频。查看单条详情时才读取分段 BLOB 并返回为 `audio_wav_base64`。

## API

- `POST /api/records`：新建记录。
- `GET /api/records?q=关键词`：列出或搜索当前用户的记录。
- `GET /api/records/{record_id}`：读取完整记录和分段音频。
- `PUT /api/records/{record_id}`：保存继续整理后的记录。
- `DELETE /api/records/{record_id}`：删除单条记录。
- `GET /api/records/{record_id}/text`：从数据库导出纯文本。

这些接口沿用当前账号鉴权；管理员可以查看全部本地记录，普通用户只能访问自己的记录。

## 界面流程

1. 在会议转写页点击“新建记录”。
2. 使用麦克风或上传音频。上传默认启用说话人分离；麦克风停止后会执行最终说话人分离。
3. 修改记录标题并点击“保存记录”。
4. 在“记录管理”中搜索、查看、载入、删除或导出历史记录。

快捷键 `Ctrl+H` 可打开或关闭记录管理面板。
