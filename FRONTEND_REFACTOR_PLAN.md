# DiTing v4.5 前端布局重构方案

> 本文档详细描述前端布局的 5 项改动，涉及 `index.html`、`css/styles.css`、`js/app.js` 三个文件。

---

## 改动总览

| # | 改动 | 影响区域 |
|---|------|---------|
| A | 删除会议领域卡片 + 删除中下方摘要横栏 + 删除右栏逻辑校验 | 底部结果面板 + 右侧边栏 |
| B | 合并"处理状态"和"统计"为一栏，顶部加圆形进度环 | 右侧边栏 |
| C | 底部结果面板从 4 列改为 4 列重新分配：关键术语 + 说话人分布 + 会议摘要 + 行动项 | 底部结果面板 |
| D | 每条 seg 的播放按钮改为图标按钮，移至时长右侧 | 转录面板 |
| E | 原播放按钮位置改为可展开波形图栏位 | 转录面板 |

---

## 改动 A — 删除三个区域

### A1. 删除"会议领域"卡片

**文件**：`frontend/index.html` 第 109-115 行

删除整个领域卡片：
```html
<!-- 删除以下内容 -->
<div class="results-card">
  <div class="results-card-header">会议领域</div>
  <div class="results-card-body" id="domainCard">
    <div class="summary-empty">分析中…</div>
  </div>
</div>
```

**文件**：`frontend/js/app.js`

- 删除 `const domainCard = $('domainCard');` (第 62 行)
- `resetResultsPanel()` 中删除 `domainCard` 相关行 (第 659 行)
- `renderResultsPanel()` 中删除领域渲染分支 (第 668-673 行)
- `renderDomainCard()` 函数可保留但不再被调用，或直接删除 (第 694-728 行)

### A2. 删除中下方"会议处理完成后将生成摘要"横栏

**文件**：`frontend/index.html` 第 97-102 行

删除：
```html
<!-- 删除以下内容 -->
<div class="summary-panel" id="summaryPanel">
  <div style="font-size:12px;color:var(--text-muted);text-align:center;">
    会议处理完成后将生成摘要...
  </div>
</div>
```

**文件**：`frontend/js/app.js`

- 删除 `const summaryPanel = $('summaryPanel');` (第 55 行)
- `resetAll()` 中删除 `summaryPanel` 相关行 (第 1787 行)
- 同时删除 `.summary-panel` 对应的 CSS (styles.css 第 241-249 行)

### A3. 删除右侧"逻辑校验"卡片

**文件**：`frontend/index.html` 第 89-94 行

删除：
```html
<!-- 删除以下内容 -->
<div class="sidebar-card">
  <div class="sidebar-card-header">逻辑校验</div>
  <div class="sidebar-card-body" id="logicPanel">
    <div style="font-size:11px;color:var(--text-muted);text-align:center;padding:10px 0;">暂无数据</div>
  </div>
</div>
```

**文件**：`frontend/js/app.js`

- 删除 `const logicPanel = $('logicPanel');` (第 54 行)
- `resetAll()` 中删除 `logicPanel` 相关行 (第 1785 行)
- `updateLogicPanel()` 函数不再被渲染，可保留或删除 (第 631-650 行)

---

## 改动 B — 合并"处理状态"和"统计"为一栏

### B1. HTML 结构

**文件**：`frontend/index.html`

将原来独立的"统计"卡片和"处理状态"卡片合并为一个卡片，替换原第 52-87 行的内容：

```html
<!-- Right Sidebar -->
<div class="sidebar">
  <div class="sidebar-card">
    <div class="sidebar-card-header">处理状态 & 统计</div>
    <div class="sidebar-card-body">

      <!-- 圆形进度环 -->
      <div class="progress-ring-container">
        <svg class="progress-ring" width="80" height="80">
          <circle class="progress-ring-bg" cx="40" cy="40" r="34" />
          <circle class="progress-ring-fg" cx="40" cy="40" r="34" id="progressRingFg" />
        </svg>
        <div class="progress-ring-text">
          <span class="progress-ring-pct" id="progressRingPct">0%</span>
          <span class="progress-ring-stage" id="progressRingStage">等待</span>
        </div>
      </div>

      <!-- 处理阶段 -->
      <div class="status-stage-row" id="processingPanel">
        <div style="font-size:11px;color:var(--text-muted);text-align:center;padding:6px 0;">等待输入...</div>
      </div>

      <!-- 统计数据 -->
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-value" style="color:var(--accent-blue);" id="statSegments">0</div>
          <div class="stat-label">分段</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" style="color:var(--accent-orange);" id="statLogicFlags">0</div>
          <div class="stat-label">逻辑提示</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" style="color:var(--accent-red);" id="statLowConf">0</div>
          <div class="stat-label">低置信</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" style="color:var(--accent-green);" id="statCorrections">0</div>
          <div class="stat-label">纠错</div>
        </div>
      </div>

    </div>
  </div>
</div>
```

### B2. CSS 样式

**文件**：`frontend/css/styles.css`

新增进度环样式（追加到 `.stats-grid` 之前）：

```css
/* ── Progress Ring ── */
.progress-ring-container {
  position: relative;
  width: 80px;
  height: 80px;
  margin: 0 auto 10px;
}
.progress-ring {
  transform: rotate(-90deg);
}
.progress-ring-bg {
  fill: none;
  stroke: var(--bg-tertiary);
  stroke-width: 5;
}
.progress-ring-fg {
  fill: none;
  stroke: var(--accent-blue);
  stroke-width: 5;
  stroke-linecap: round;
  stroke-dasharray: 213.628;  /* 2 * PI * 34 */
  stroke-dashoffset: 213.628; /* 初始为空 */
  transition: stroke-dashoffset 0.5s ease, stroke 0.3s ease;
}
.progress-ring-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}
.progress-ring-pct {
  display: block;
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
}
.progress-ring-stage {
  display: block;
  font-size: 9px;
  color: var(--text-muted);
  margin-top: 1px;
}
.status-stage-row {
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
```

### B3. JS 逻辑

**文件**：`frontend/js/app.js`

新增进度环更新函数：

```javascript
// ── Progress Ring ──
const RING_CIRCUMFERENCE = 2 * Math.PI * 34; // ≈ 213.628

function updateProgressRing(fraction, stageLabel) {
  const ringFg = $('progressRingFg');
  const ringPct = $('progressRingPct');
  const ringStage = $('progressRingStage');
  if (!ringFg) return;

  const pct = Math.max(0, Math.min(1, fraction));
  ringFg.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - pct));

  // 颜色随进度变化
  if (pct >= 1) {
    ringFg.style.stroke = 'var(--accent-green)';
  } else if (pct > 0.5) {
    ringFg.style.stroke = 'var(--accent-blue)';
  } else {
    ringFg.style.stroke = 'var(--accent-orange)';
  }

  if (ringPct) ringPct.textContent = Math.round(pct * 100) + '%';
  if (ringStage && stageLabel) ringStage.textContent = stageLabel;
}
```

需要修改 `processingPanel.innerHTML` 赋值的地方（第 1028 行和第 1136 行），在更新阶段文本的同时调用 `updateProgressRing()`：

```javascript
// 在 upload progress 更新中追加：
updateProgressRing(fraction, stage);
```

在 `resetAll()` 中重置进度环：
```javascript
updateProgressRing(0, '等待');
```

---

## 改动 C — 底部结果面板重新分配为 4 列

### C1. HTML 结构

**文件**：`frontend/index.html`

底部 `results-panel` 改为 4 列等宽布局，内容为：关键术语 + 说话人分布 + 会议摘要 + 行动项

```html
<div class="results-panel" id="resultsPanel">
  <!-- 关键术语 -->
  <div class="results-card">
    <div class="results-card-header">关键术语</div>
    <div class="results-card-body" id="hotwordCard">
      <div class="summary-empty">提取中…</div>
    </div>
  </div>

  <!-- 说话人分布 -->
  <div class="results-card">
    <div class="results-card-header">说话人分布</div>
    <div class="results-card-body" id="speakerCard">
      <div class="summary-empty">识别中…</div>
    </div>
  </div>

  <!-- 会议摘要 -->
  <div class="results-card">
    <div class="results-card-header">会议摘要</div>
    <div class="results-card-body" id="summaryCard">
      <div class="summary-empty">会议处理完成后将自动生成摘要…</div>
    </div>
  </div>

  <!-- 行动项 / TODO -->
  <div class="results-card">
    <div class="results-card-header">行动项 / TODO</div>
    <div class="results-card-body" id="actionCard">
      <div class="summary-empty">暂无行动项</div>
    </div>
  </div>
</div>
```

### C2. CSS 调整

**文件**：`frontend/css/styles.css`

`.results-panel` 的 `grid-template-columns` 改为等宽 4 列：

```css
.results-panel {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;  /* 原: 1fr 1fr 1fr 2fr → 改为等宽 */
  gap: 10px;
  max-height: 210px; overflow: hidden;
}
```

### C3. JS 逻辑

**文件**：`frontend/js/app.js`

新增 `actionCard` 引用：
```javascript
const actionCard = $('actionCard');
```

修改 `renderSummaryCard()` — 只渲染摘要文本和主题/决策，不再渲染行动项（行动项移到独立卡片）：

```javascript
function renderSummaryCard(summary) {
  const card = $('summaryCard');
  if (!card) return;

  const hasContent = (summary.summary && summary.summary.trim()) ||
    (summary.topics && summary.topics.length > 0) ||
    (summary.decisions && summary.decisions.length > 0);

  if (!hasContent) {
    card.innerHTML = '<div class="summary-empty">暂未生成摘要</div>';
    return;
  }

  let h = '';
  if (summary.summary && summary.summary.trim()) {
    h += `<div class="summary-text">${escapeHTML(summary.summary)}</div>`;
  }
  if (summary.topics && summary.topics.length > 0) {
    h += '<div class="summary-section"><div class="summary-section-title">讨论主题</div>';
    summary.topics.forEach(t => {
      const text = typeof t === 'string' ? t : (t.topic || t);
      h += `<span class="summary-topic-tag">${escapeHTML(text)}</span>`;
    });
    h += '</div>';
  }
  if (summary.decisions && summary.decisions.length > 0) {
    h += '<div class="summary-section"><div class="summary-section-title">会议决策</div>';
    summary.decisions.forEach(d => {
      const text = typeof d === 'string' ? d : (d.decision || '');
      h += `<div class="summary-action-item" style="border-left:2px solid var(--accent-yellow);padding-left:8px;">${escapeHTML(text)}</div>`;
    });
    h += '</div>';
  }
  card.innerHTML = h;
}
```

修改 `fetchAndRenderActionItems()` — 渲染到独立的 `actionCard` 而非追加到 `summaryCard`：

```javascript
async function fetchAndRenderActionItems(segments) {
  if (!segments || segments.length === 0) return;
  const card = $('actionCard');  // ← 改为 actionCard
  if (!card) return;

  // ... fetch 逻辑不变 ...

  if (data.action_items && data.action_items.length > 0) {
    let h = '';
    data.action_items.forEach((a) => {
      const pColor = a.priority === 'high' ? '#f85149' : a.priority === 'low' ? '#8b949e' : '#58a6ff';
      const pLabel = a.priority === 'high' ? '高' : a.priority === 'low' ? '低' : '中';
      h += `<div style="margin:6px 0;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;border-left:3px solid ${pColor};">`;
      h += `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">`;
      h += `<span style="font-size:11px;padding:2px 6px;border-radius:3px;background:${pColor}33;color:${pColor};font-weight:600;">${pLabel}</span>`;
      h += `<span style="font-size:13px;color:#e6edf3;font-weight:500;">${escapeHTML(a.task || '')}</span>`;
      h += `</div>`;
      const meta = [];
      if (a.assignee) meta.push(`👤 ${escapeHTML(a.assignee)}`);
      if (a.deadline) meta.push(`⏰ ${escapeHTML(a.deadline)}`);
      if (a.speaker) meta.push(`🎙 ${escapeHTML(a.speaker)}`);
      if (meta.length) h += `<div style="margin-top:4px;font-size:11px;color:#8b949e;display:flex;gap:12px;flex-wrap:wrap;">${meta.join('')}</div>`;
      if (a.source_text) h += `<div style="margin-top:6px;font-size:11px;color:#6e7681;border-top:1px solid #21262d;padding-top:4px;font-style:italic;">"${escapeHTML(a.source_text)}"</div>`;
      h += `</div>`;
    });
    card.innerHTML = h;
  } else {
    card.innerHTML = '<div class="summary-empty">暂无行动项</div>';
  }
}
```

同时如果 `summary.action_items` 也存在（演示模式），将其渲染到 `actionCard`：

```javascript
function renderSummaryCardActions(summary) {
  const card = $('actionCard');
  if (!card) return;
  if (summary.action_items && summary.action_items.length > 0) {
    let h = '';
    summary.action_items.forEach((a) => {
      const item = typeof a === 'string' ? { task: a } : a;
      // ... 同上渲染逻辑 ...
    });
    card.innerHTML = h;
  }
}
```

在 `renderResultsPanel()` 中调用：
```javascript
if (data.summary && data.summary.action_items) {
  renderSummaryCardActions(data.summary);
}
```

修改 `resetResultsPanel()` 追加 `actionCard` 重置：
```javascript
function resetResultsPanel() {
  if (hotwordCard) hotwordCard.innerHTML = '<div class="summary-empty">提取中…</div>';
  if (speakerCard) speakerCard.innerHTML = '<div class="summary-empty">识别中…</div>';
  if (summaryCard) summaryCard.innerHTML = '<div class="summary-empty">会议处理完成后将自动生成摘要…</div>';
  if (actionCard) actionCard.innerHTML = '<div class="summary-empty">暂无行动项</div>';
}
```

修改 `resetAll()` 追加 `actionCard` 重置。

---

## 改动 D — 播放按钮改为图标按钮

### D1. 新的按钮结构

**文件**：`frontend/js/app.js` 的 `renderSegment()` 函数（第 538-594 行）

将播放按钮从文本"播放"改为橘色圆底板 + 白色三角形图标，位置从每条 seg 的最右端移到时长右侧：

**原结构**（第 570-576 行）：
```javascript
card.innerHTML = `
  <div class="segment-top">
    <span class="segment-index">#${index + 1}</span>
    <span class="segment-dur">时长 ${durationText}</span>
    <span class="segment-confidence" style="...">置信度 ...</span>
    ${hasAudio ? `<button class="segment-play-btn" ...>播放</button>` : ''}
  </div>
  <div class="segment-text">${textHtml}</div>
`;
```

**新结构**：
```javascript
card.innerHTML = `
  <div class="segment-top">
    <span class="segment-index">#${index + 1}</span>
    <span class="segment-dur">时长 ${durationText}</span>
    ${hasAudio ? `<button class="seg-play-icon" data-seg-index="${index}" type="button" title="播放/暂停">
      <svg width="10" height="10" viewBox="0 0 10 10"><polygon points="2,1 9,5 2,9" fill="white"/></svg>
    </button>` : ''}
    <span class="segment-confidence" style="color:${confColor};font-weight:600;">置信度 ${confidenceText}${confLabel ? ' (' + confLabel + ')' : ''}</span>
    <span class="seg-expand-toggle" data-seg-index="${index}" title="展开/收起波形">▾</span>
  </div>
  <div class="segment-text">${textHtml}</div>
  <div class="seg-waveform-expand" id="segWaveform_${index}" style="display:none;">
    <div class="seg-waveform-container" id="segWaveContainer_${index}"></div>
  </div>
`;
```

关键变化：
1. 播放按钮使用 `.seg-play-icon` 类（替代 `.segment-play-btn`），内含 SVG 三角形
2. 位置移到 `segment-dur` 之后、`segment-confidence` 之前
3. 原播放按钮位置替换为 `.seg-expand-toggle`（展开/收起波形图）
4. 新增 `.seg-waveform-expand` 容器（默认隐藏）

### D2. CSS 样式

**文件**：`frontend/css/styles.css`

删除旧的 `.segment-play-btn` 样式（第 127-144 行），新增：

```css
/* ── Segment Play Icon Button ── */
.seg-play-icon {
  width: 22px;
  height: 22px;
  border: none;
  border-radius: 50%;
  background: var(--accent-orange);
  color: white;
  font-size: 0;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
  padding: 0;
  line-height: 0;
}
.seg-play-icon:hover {
  opacity: 0.85;
  transform: scale(1.08);
}
.seg-play-icon.playing {
  background: var(--accent-green);
}
.seg-play-icon svg {
  pointer-events: none;
}

/* ── Segment Expand Toggle ── */
.seg-expand-toggle {
  font-size: 12px;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0 4px;
  transition: color 0.2s;
  min-width: 20px;
  text-align: center;
}
.seg-expand-toggle:hover {
  color: var(--text-primary);
}
.seg-expand-toggle.expanded {
  transform: rotate(180deg);
  display: inline-block;
}

/* ── Segment Waveform Expand ── */
.seg-waveform-expand {
  padding: 8px 10px;
  border-top: 1px solid var(--border);
  background: var(--bg-tertiary);
}
.seg-waveform-container {
  width: 100%;
  min-height: 50px;
}
```

### D3. JS 事件绑定

**文件**：`frontend/js/app.js` 的 `renderSegment()` 函数

修改事件绑定部分：

```javascript
if (hasAudio) {
  audioManager.createAudio(index, audioBase64);

  // 播放按钮点击
  const playBtn = card.querySelector(`.seg-play-icon[data-seg-index="${index}"]`);
  if (playBtn) {
    playBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (audioManager.activeIndex === index) audioManager.pause();
      else audioManager.play(index);
    });
  }

  // 展开波形图
  const expandToggle = card.querySelector(`.seg-expand-toggle[data-seg-index="${index}"]`);
  const waveformExpand = card.querySelector(`#segWaveform_${index}`);
  const waveContainer = card.querySelector(`#segWaveContainer_${index}`);

  if (expandToggle && waveformExpand) {
    let waveCreated = false;
    expandToggle.addEventListener('click', async (e) => {
      e.stopPropagation();
      const isHidden = waveformExpand.style.display === 'none';
      if (isHidden) {
        waveformExpand.style.display = 'block';
        expandToggle.classList.add('expanded');
        // 懒加载波形图
        if (!waveCreated && waveContainer && typeof WaveSurfer !== 'undefined') {
          await audioManager.create(index, audioBase64, waveContainer);
          waveCreated = true;
        }
      } else {
        waveformExpand.style.display = 'none';
        expandToggle.classList.remove('expanded');
      }
    });
  }
}
```

### D4. 高亮逻辑更新

**文件**：`frontend/js/app.js` 的 `_highlightSegment()` 方法（第 418-429 行）

将选择器从 `.segment-play-btn` 改为 `.seg-play-icon`：

```javascript
_highlightSegment(segIndex, on) {
  const el = document.querySelector(`.segment[data-seg-index="${segIndex}"]`);
  if (el) {
    if (on) el.classList.add('playing');
    else el.classList.remove('playing');
  }
  const btn = document.querySelector(`.seg-play-icon[data-seg-index="${segIndex}"]`);
  if (btn) {
    if (on) btn.classList.add('playing');
    else btn.classList.remove('playing');
  }
}
```

---

## 改动 E — 可展开波形图栏位

### E1. 设计说明

每条 seg 卡片的结构变为：

```
┌──────────────────────────────────────────────────────┐
│ #1  时长 0:05  ▶(橘色圆)  置信度 92% (高)      ▾  │  ← segment-top 行
├──────────────────────────────────────────────────────┤
│ 转写文本内容...                                       │  ← segment-text
├──────────────────────────────────────────────────────┤
│ [波形图 — 仅展开时显示]                              │  ← seg-waveform-expand
│ ┌────────────────────────────────────────┐           │
│ │ ▁▂▃▅▇█▇▅▃▂▁▂▃▅▇█▇▅▃▂▁              │           │
│ └────────────────────────────────────────┘           │
│ 0:00                                    0:05         │
└──────────────────────────────────────────────────────┘
```

- 默认只显示文本内容，不显示波形图
- 点击 `▾` 图标展开，显示该 seg 的波形图
- 波形图使用 WaveSurfer.js，支持拖动跳转到该段音频的任意时间点
- 再次点击 `▾` 收起波形图
- `.seg-expand-toggle` 旋转 180° 表示展开状态

### E2. 波形图懒加载策略

波形图实例在首次展开时才创建，避免一次性创建大量 WaveSurfer 实例导致性能问题：

```javascript
// 展开时调用 audioManager.create() (非 createAudio)
// audioManager.create() 会创建带波形的 WaveSurfer 实例
// 同时暂停并销毁之前的 createAudio() 创建的轻量 Audio 实例
```

注意：`audioManager.create()` 已经支持 `dragToSeek: true`（第 144 行），用户拖动波形图即可跳转到该段音频的任意位置。

### E3. 同时存在的音频实例处理

当展开波形图后，播放控制切换到 WaveSurfer 实例（`entry.ws`），而非之前的 HTMLAudioElement（`entry.audio`）。`AudioPlaybackManager` 已支持双模式：

```javascript
// play() 方法已处理：
if (entry.ws) entry.ws.play();      // 优先用 WaveSurfer
else if (entry.audio) entry.audio.play();  // 退回 HTMLAudio
```

但需要修改 `create()` 方法在已有 `createAudio()` 实例时的处理：

```javascript
async create(segIndex, audioBase64, containerEl) {
  // 如果已有 audio 实例，先销毁
  const existing = this.instances.get(segIndex);
  if (existing && existing.audio && !existing.ws) {
    try {
      existing.audio.pause();
      existing.audio.src = '';
    } catch (e) { /* ignore */ }
    try { URL.revokeObjectURL(existing.blobUrl); } catch (e) { /* ignore */ }
    this.instances.delete(segIndex);
  }
  // ... 继续 WaveSurfer 创建逻辑 ...
}
```

### E4. 收起时清理

收起波形图时不需要销毁 WaveSurfer 实例（保留缓存），只是隐藏 DOM：

```javascript
// 收起时不调用 destroy，只隐藏
waveformExpand.style.display = 'none';
expandToggle.classList.remove('expanded');
// 如果正在播放该段，不暂停，只是视觉收起
```

但在 `resetAll()` 中会调用 `audioManager.reset()` 清理所有实例。

---

## 改动文件清单

| 文件 | 改动类型 | 行数变化 |
|------|---------|---------|
| `frontend/index.html` | 删除领域卡片、摘要横栏、逻辑校验卡片；重构侧栏和结果面板 | -15 行, +35 行 |
| `frontend/css/styles.css` | 删除旧播放按钮和摘要面板样式；新增进度环、图标按钮、展开波形样式 | -20 行, +70 行 |
| `frontend/js/app.js` | 删除领域/逻辑相关引用；新增进度环、actionCard、图标按钮、波形展开逻辑 | -30 行, +80 行 |

---

## 验证清单

### 布局验证
- [ ] 会议领域卡片已删除
- [ ] 中下方"会议处理完成后将生成摘要"横栏已删除
- [ ] 右栏逻辑校验已删除
- [ ] 处理状态和统计合为一栏
- [ ] 进度环正确显示百分比和阶段
- [ ] 底部 4 列等宽：关键术语、说话人分布、会议摘要、行动项

### 功能验证
- [ ] 演示模式正常
- [ ] 上传音频后结果面板正确渲染
- [ ] 行动项渲染到独立卡片
- [ ] 每条 seg 的播放按钮为橘色圆形图标
- [ ] 播放按钮位于时长右侧
- [ ] 点击播放按钮可播放/暂停
- [ ] 播放中按钮变为绿色
- [ ] 点击 ▾ 展开波形图
- [ ] 波形图可拖动跳转
- [ ] 再次点击 ▾ 收起波形图
- [ ] 重置后所有面板恢复初始状态
- [ ] 麦克风录音功能正常
- [ ] 日志面板正常

### 兼容性验证
- [ ] `python start.py` 可正常启动
- [ ] 浏览器无 console 错误
- [ ] 无 MIME error
- [ ] 无 module import error
- [ ] 无 ReferenceError
```
