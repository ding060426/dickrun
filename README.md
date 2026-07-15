# 谛听 (会悟) v3.0 — Smart Meeting Speech Cognitive System

基于 X-ASR (sherpa-onnx zipformer2) 的智能会议语音认知系统。支持音频文件上传、实时转写、热词修正、逻辑校验、音频波形可视化。

## 系统架构

```
frontend/          ← 前端 (wavesurfer.js + AudioWorklet + WebSocket)
  audio-worklet.js ← 浏览器采集、真实采样率到 16 kHz 的连续降采样
backend/
  audio_buffer.py  ← ASR/VAD/diarization 共用的 16 kHz 单声道时间轴
  main.py          ← FastAPI 后端 (HTTP + WebSocket API)
  diarization/
    pipeline.py    ← 双轨分析、换人边界重识别与时间轴对齐
    sherpa_backend.py ← pyannote segmentation + 3D-Speaker 本地推理
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

项目默认使用 960ms `meeting` 模型。下载器会从官方 Hugging Face 仓库断点下载、校验并原子发布模型文件：

```powershell
python backend/xasr/download_models.py

# 也可以显式部署其他延迟档位
python backend/xasr/download_models.py --profile low-latency
```

模型目录应有以下文件：
```
backend/xasr/models/
  ├── encoder-160ms.onnx
  ├── decoder-160ms.onnx
  ├── joiner-160ms.onnx
  ├── encoder-960ms.onnx   (默认)
  ├── decoder-960ms.onnx   (默认)
  ├── joiner-960ms.onnx    (默认)
  ├── tokens.txt
  └── silero_vad.onnx       (麦克风与文件切分共用的本地 Silero VAD)
```

离线说话人日志还需要两个 sherpa-onnx 兼容模型（当前本地工作区已放置好）：

```text
backend/diarization/models/
  pyannote-segmentation-3.0.int8.onnx
  3dspeaker-eres2net.onnx
```

也可以不复制模型，直接通过环境变量指向已有文件：

```powershell
$env:DITING_DIARIZATION_SEGMENTATION_MODEL = "D:\models\pyannote\model.int8.onnx"
$env:DITING_SPEAKER_EMBEDDING_MODEL = "D:\models\3dspeaker.onnx"
```

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

如果模型未就绪，上传接口会明确返回错误，不会再用预置样例冒充真实转写。
启动器会校验后端 API revision；如果 `8765` 被旧版 会悟 占用，会明确报错并要求先关闭旧进程，
不会再把旧服务误判为本次启动成功。

### 5. 会议管理与转写结果

首页现已整合 `supabase-hero` 的会议管理系统，登录后可进入会议转写、会议预约、会议分析和用户管理。
“会议转写”直接使用当前分支的 X-ASR、Silero VAD、长音频分块、说话人分离和实时 DTP2 麦克风链路；
转写完成后可在“会议分析”中关联预约会议并保存分段、说话人、时长和质量统计。

未登录时启动页会显示账号登录界面。登录后，左上角显示当前用户头像和姓名；点击后可以查看账号摘要、
编辑显示名称、头像、邮箱和电话，或选择“切换用户”“退出登录”。切换用户会先注销当前服务端会话，
再返回登录界面，不会在浏览器中保存其他账号密码。上传的头像会先在浏览器裁剪为正方形并压缩后保存。

共享预约的使用方式：

1. 在“用户”页把账号加入自己的同事列表。
2. 在“会议预约”创建会议时勾选参会同事，并填写时间、地点和会议说明。
3. 会议会同时出现在创建者及所有参会人的个人日期界面；详情包含创建者与其他参会人。
4. 只有创建者（以及管理员）能从日期右侧栏修改时间、地点、说明和参会人；普通参会人只有查看权限。

管理数据默认保存在本地 `backend/data/diting.db`，无需联网即可使用。首次启动会创建管理员账号 `admin`，
密码读取 `DITING_ADMIN_PASSWORD`，未配置时仅为本地开发保留默认值 `admin123`。只要各账号访问同一个后端进程，
它们就共享这一个 SQLite 数据库；预约与参会人使用关系表保存，并启用外键、索引、30 秒忙等待和 WAL 并发模式。
旧数据库会在启动时自动增加 `avatar_data_url` 资料字段，不需要删除或重建数据库。

如需使用 Supabase，在 Supabase SQL Editor 中执行 `backend/supabase_init.sql`，复制并填写配置：

```powershell
Copy-Item backend/.env.example backend/.env
# 编辑 backend/.env 中的 SUPABASE_URL、SUPABASE_KEY 和管理员密码
python backend/setup_supabase.py
python start.py
```

只有 `SUPABASE_URL` 与 `SUPABASE_KEY` 同时存在时才启用 Supabase；否则自动回退到本地 SQLite，
不会影响本地转写服务启动。Supabase 初始化脚本会迁移旧版 JSON 参会人数据，并用 PostgreSQL 触发器在同一事务中
同步预约和参会关系。当前 SQL 中的 RLS 策略仅用于开发调试；推向公网前必须把 `SUPABASE_KEY` 保留在后端，
使用服务端密钥，并将开发阶段的全开放策略替换为正式最小权限策略。

### 6. 离线多说话人会议

上传前勾选底部“说话人”，人数已知时选择 `2～8 人`；已知人数会直接约束聚类，通常比自动估计稳定。
后端会让 diarization 与 X-ASR 共享同一份标准音频并独立分析，在换人点对跨说话人的 ASR 长段做带边界留白的
局部重识别，最后返回 `speaker_id`、`speaker_confidence`、`overlap` 和时间戳。点击彩色说话人标签可重命名，
当前任务中的同一标签会同步更新。

HTTP 上传参数：

```text
POST /api/audio/upload?enable_diarization=true&num_speakers=4
PATCH /api/meetings/{meeting_id}/speakers/SPEAKER_00  {"name":"张三"}
```

模型缺失或显式关闭时会安全降级为原来的纯 ASR，不会让文件上传失败。重叠语音第一版只标记和降低归属置信度，
不会把混合语音伪装成已经分离的双路文本。

超过 10 分钟的音频默认启用静音感知分块：目标块长 5 分钟、最大 8 分钟、前后保留 2 秒上下文，最多两个
worker 并行执行说话人分析。每个块先生成本地说话人，再通过 3D-Speaker 声纹做全局聚类，因此不会直接把
不同块中的 `SPEAKER_00` 当成同一个人。X-ASR 仍保留整段连续识别上下文，不随 diarization 分块。
前端处理状态会显示 `diarization 2/6` 之类的块级进度。

长音频参数可通过环境变量调整：

```text
DITING_DIARIZATION_CHUNKING=true
DITING_DIARIZATION_LONG_AUDIO_SEC=600
DITING_DIARIZATION_TARGET_CHUNK_SEC=300
DITING_DIARIZATION_MAX_CHUNK_SEC=480
DITING_DIARIZATION_CHUNK_OVERLAP_SEC=2
DITING_DIARIZATION_MAX_WORKERS=2
DITING_DIARIZATION_WORKER_THREADS=2
DITING_SPEAKER_STITCH_THRESHOLD=0.75
```

单块失败会使用全新 worker 重试；分块或跨块声纹统一仍失败时回退到整段说话人分析，整段也失败才降级为纯 ASR。

### 7. 实时麦克风转写

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

点击底部 `Settings` 可在同一界面调整识别、麦克风和热词。识别设置支持为实时预览与
最终转写分别选择模型档位（默认均为 960ms），并调整本地 Silero 文件切分阈值、最短
语音/静音与前后留白。麦克风设置支持枚举设备、录音电平测试、回声消除、降噪、自动增益、
VAD 门控、pre-roll 和句尾宽限期。运行设置保存到 `backend/data/settings.json`。

热词设置支持启用/停用解码增强、调整全局默认权重、逐词设置权重，以及开启模糊拼音纠错。
配置保存到本机 `backend/data/hotwords.json`，从下一次麦克风或文件识别会话开始生效。模糊拼音只对
至少两个汉字的中文热词执行最长优先匹配，支持平翘舌、`n/l` 和前后鼻音归一化；
英文热词还会统一大小写、空格、缩写和常见误识别写法。

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
`quality`（1920ms），对应 ONNX 文件存在时才可启用。默认的实时与最终转写都使用
`meeting`（960ms）；两者相同时共享一份已预热 ONNX 运行时，避免重复占用内存。
识别、VAD 与麦克风参数通过前端 `Settings` 调整；服务容量相关参数仍保留环境变量入口：

```powershell
$env:DITING_MAX_UPLOAD_MB = "2048"
$env:DITING_PROCESSING_WORKERS = "2"
python start.py
```

如已安装 `torch` 与 `pycorrector`，可显式启用 tep 分支同步来的 MacBERT
同音字/形似字纠错。默认关闭，避免首次加载模型阻塞最终转写：

```powershell
$env:DITING_ENABLE_MACBERT = "1"
python start.py
```

无需物理麦克风也可以用音频文件验证同一条 WebSocket 链路：

```bash
python backend/tests/smoke_live_websocket.py path/to/meeting.wav --url ws://127.0.0.1:8765/ws/live
```

若要用自己的标注音频对比 160ms 与 960ms 的原始 CER、热词召回率和运行时间，可创建：

```json
[
  {"audio": "sample.wav", "reference": "这是参考文本", "keywords": ["参考文本"]}
]
```

然后运行：

```powershell
python backend/evaluate_profiles.py manifest.json --profiles low-latency meeting --output report.json
```

## 功能

| 功能 | 状态 | 说明 |
|------|------|------|
| ASR 转写 | ✅ | sherpa-onnx zipformer2 流式推理 |
| VAD + 端点检测 | ✅ | 文件：Silero 提供时间锚点，X-ASR 跨段保留上下文并按完整句合并；实时：Silero 优先、能量 VAD 降级 + 可配置 pre-roll |
| 热词修正 | ✅ | 可持久化逐词权重 + modified beam search + 多音字模糊拼音 + ASCII 标准化 |
| 文本后处理 | ✅ | 保守去口癖、标点恢复、重复清理与中文数字 ITN |
| 逻辑校验 | ✅ | 数据冲突检测 (数字/百分比对比) |
| 不确定性估计 | ✅ | 低置信度区段标记 |
| 音频波形可视化 | ✅ | wavesurfer.js + WAV 编码 |
| 实时录音转写 | ✅ | AudioWorklet + DTP2 帧序号 + partial/final + 持久 WAV + 停止后二次定稿和离线说话人对齐 |
| Eval_Ali 评测 | ✅ | CER 计算 / 热词提取 |
| 说话人日志/分段 | ✅ | pyannote segmentation + 3D-Speaker，全局聚类、时间平滑、边界重识别与可重命名标签 |
| 重叠语音双路分离 | 🔜 | LocalMeet 已有 MossFormer2 离线 GPU 实现；不放入低延迟实时预览主链路 |
| 说话人识别 | 🔜 | 声纹注册库匹配 待实现 |

## License

MIT
