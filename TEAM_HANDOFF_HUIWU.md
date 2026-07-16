# HUIWU 集成说明

更新时间：2026-07-16

## 1. 项目位置

- 集成目录：`D:\HUIWU`
- 基线来源：`D:\diting4.5`
- 对接来源分支：`https://github.com/ding060426/dickrun/tree/agent/realtime-mic-visualizer`
- 启动入口：`start.py`

## 2. 启动方式

```bat
cd /d D:\HUIWU
python start.py
```

启动后访问：

- 前端：`http://localhost:3000`
- 后端：`http://localhost:8765`
- 健康检查：`http://localhost:8765/api/health`
- 历史记忆：`http://localhost:8765/api/memory/history`

## 3. 本轮集成内容

### 3.1 保留 DiTing 4.5 基线能力

- 上传音频/视频识别；
- 上传进度 WebSocket；
- 本地 SQLite 历史记忆；
- 会议摘要 API 与规则 fallback；
- X-ASR 模型加载；
- 前端历史面板、摘要、热词、说话人分布展示。

### 3.2 新增实时麦克风增强

后端新增模块：

- `backend/modules/dtp2_protocol.py`
- `backend/modules/pcm_utils.py`
- `backend/modules/realtime_vad.py`
- `backend/modules/realtime_session.py`
- `backend/modules/rnnoise_filter.py`

`/ws/live` 现在支持：

1. 原有 JSON/base64 Float32 协议；
2. 新增 DTP2 binary PCM 协议；
3. DTP2 start/stop 控制消息；
4. 实时 VAD 状态；
5. RNNoise 状态字段；
6. 协议异常 warning，不直接断开。

### 3.3 新增前端辅助模块

从组员分支摘取并放入：

- `frontend/audio-worklet.js`
- `frontend/live-protocol.js`
- `frontend/mic-level.js`
- `frontend/app-settings.js`

前端麦克风路径优先级：

```text
AudioWorklet + DTP2 PCM
-> ScriptProcessor + DTP2 PCM
-> ScriptProcessor + JSON/base64 Float32 fallback
```

### 3.4 RNNoise 降噪

新增 `backend/modules/rnnoise_filter.py`，采用 `ctypes` 尝试加载预编译 `rnnoise.dll`。

默认查找：

- `backend/third_party/rnnoise/rnnoise.dll`
- 环境变量 `RNNOISE_DLL_PATH`
- 系统 PATH

如果 DLL 不存在或加载失败：

- 后端不报错；
- `/api/health` 中显示 `rnnoise.available=false`；
- ASR 继续使用原始音频。

可选环境变量：

```env
DITING_RNNOISE_ENABLED=auto
DITING_RNNOISE_LIVE_ENABLED=off
RNNOISE_DLL_PATH=D:\HUIWU\backend\third_party\rnnoise\rnnoise.dll
```

说明：当前未把 RNNoise 源码编译流程强行加入 requirements，避免 Windows 用户环境编译失败导致项目无法启动。

## 4. 已验证项目

已完成：

- Python 修改文件语法编译通过；
- 前端 module JS 语法检查通过；
- `python start.py` 启动成功；
- `GET /api/health` 返回 200；
- `GET /api/memory/history` 返回 200；
- `GET http://localhost:3000` 返回 200；
- `/ws/live` DTP2 start + binary PCM smoke test 通过；
- `/ws/live` 旧 JSON/base64 Float32 smoke test 通过。

## 5. 后续实验建议

建议继续按以下维度实际录音对比：

- 安静短句；
- 长句；
- 句中停顿；
- 背景噪声；
- 低音量；
- 中英混合热词；
- 静音；
- 连续开始/停止。

记录：

- 是否报错；
- 首字延迟；
- final 延迟；
- 分句数量；
- 空 partial 次数；
- 前端控制台错误；
- WebSocket 断连次数；
- 主观识别质量。

## 6. 注意事项

1. 不要整文件覆盖 `backend/main.py` 或 `frontend/index.html`。
2. RNNoise DLL 是可选增强，不是项目启动硬依赖。
3. 本地 `.env`、数据库、日志、音视频文件仍不应提交。
4. 如果后续要真正启用 RNNoise，需要准备 Windows x64 预编译 DLL，并附带许可证说明。
