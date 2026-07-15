// ============================================================
// DiTing v4.5 - Frontend with wavesurfer.js
// ============================================================

// ── Import wavesurfer.js (non-blocking — don't block UI init) ──
let WaveSurfer = null;
let HAS_WAVESURFER = false;

// Load wavesurfer.js asynchronously — do NOT block the UI from initializing.
// If the CDN is unreachable or slow, the page still works fully.
(async function loadWaveSurfer() {
  try {
    // Race against a 5-second timeout so we don't wait forever
    const importPromise = import('https://unpkg.com/wavesurfer.js@7.8/dist/wavesurfer.esm.js');
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('CDN timeout')), 5000)
    );
    const wsModule = await Promise.race([importPromise, timeoutPromise]);
    WaveSurfer = wsModule.default;
    HAS_WAVESURFER = true;
    console.log('[DiTing] WaveSurfer.js loaded');
  } catch (e) {
    console.warn('[DiTing] WaveSurfer.js unavailable — waveform disabled:', e.message);
    WaveSurfer = null;
    HAS_WAVESURFER = false;
  }
})();

// ── Constants ────────────────────────────────────────────────
const API_BASE = 'http://localhost:8765';
const UPLOAD_TASK_STORAGE_KEY = 'diting:lastUploadTask';

// ── DOM refs ─────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const transcriptBody = $('transcriptBody');
const emptyState = $('emptyState');
const globalAudioBar = $('globalAudioBar');
const audioPulse = $('audioPulse');
const audioStatusText = $('audioStatusText');
const audioTimeDisplay = $('audioTimeDisplay');
const segmentCounter = $('segmentCounter');
const progressFill = $('progressFill');
const timeDisplay = $('timeDisplay');
const meetingTitleDisplay = $('meetingTitleDisplay');
const connectionStatus = $('connectionStatus');
const xasrStatus = $('xasrStatus');
const statSegments = $('statSegments');
const statLogicFlags = $('statLogicFlags');
const statLowConf = $('statLowConf');
const statCorrections = $('statCorrections');
const overallConfBar = $('overallConfBar');
const overallConfText = $('overallConfText');
const processingPanel = $('processingPanel');
const logicPanel = $('logicPanel');
const summaryPanel = $('summaryPanel');
const micStatus = $('micStatus');
const micOrbPanel = $('micOrbPanel');
const micOrbWrap = $('micOrbWrap');
const micOrbWave = $('micOrbWave');
const micOrbTitle = $('micOrbTitle');
const micOrbSubtitle = $('micOrbSubtitle');
const domainCard = $('domainCard');
const hotwordCard = $('hotwordCard');
const speakerCard = $('speakerCard');
const summaryCard = $('summaryCard');

// ── State ────────────────────────────────────────────────────
let allSegments = [];
let currentSegmentIndex = -1;
let isPlaying = false;
let playbackTimer = null;
let demoData = null;

// Upload
let uploadWs = null;
let uploadFileId = null;
let uploadHttpResult = null;   // fallback: HTTP response when WS doesn't deliver
let uploadWsDelivered = false; // whether WS delivered any segments
let uploadAbortController = null;
let uploadFallbackTimer = null;
let uploadCloseFallbackTimer = null;
let resetGeneration = 0;

// Live mic
let micStream = null;
let micWs = null;
let micAudioContext = null;
let micAnalyser = null;
let micVisualizerRAF = null;
let micLevelSmoothed = 0;
let isRecording = false;

// Log stream
let logWs = null;
let logVisible = false;

// ═══════════════════════════════════════════════════════════
// AudioPlaybackManager (wavesurfer.js based)
// ═══════════════════════════════════════════════════════════
class AudioPlaybackManager {
  constructor() {
    /** @type {Map<number, {ws: object, blobUrl: string, duration: number}>} */
    this.instances = new Map();
    /** @type {number | null} */
    this.activeIndex = null;
    /** @type {number | null} */
    this.dBUpdateRAF = null;
    this.maxInstances = 40;  // soft cap — destroy oldest when exceeded
  }

  /**
   * Convert base64 WAV to a Blob URL and create a WaveSurfer instance.
   */
  async create(segIndex, audioBase64, containerEl) {
    if (!HAS_WAVESURFER || !audioBase64) return null;

    // Destroy existing instance for this index (shouldn't happen, but be safe)
    this.destroy(segIndex);

    // Decode base64 → Blob
    let blob;
    try {
      blob = this._b64toBlob(audioBase64, 'audio/wav');
    } catch (e) {
      console.warn(`[Audio] base64 decode failed for seg ${segIndex}:`, e);
      return null;
    }

    const blobUrl = URL.createObjectURL(blob);

    // Create WaveSurfer
    const ws = WaveSurfer.create({
      container: containerEl,
      waveColor: '#3a3f55',
      progressColor: '#8b6ce8',
      cursorColor: 'rgba(139,108,232,0.5)',
      cursorWidth: 1,
      height: 44,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      interact: true,
      dragToSeek: true,
      autoScroll: false,
    });

    // Wire events
    ws.on('play', () => {
      this._onPlay(segIndex);
    });
    ws.on('pause', () => {
      this._onPause(segIndex);
    });
    ws.on('finish', () => {
      this._onFinish(segIndex);
    });
    ws.on('timeupdate', (t) => {
      this._onTimeUpdate(segIndex, t, ws.getDuration());
    });
    ws.on('ready', () => {
      console.log(`[Audio] 片段 ${segIndex} 波形已就绪，时长=${ws.getDuration().toFixed(1)}s`);
      // 不自动播放，用户点击片段或播放按钮后才播放。
    });
    ws.on('interaction', () => {
      // User clicked the waveform — play it
      this.play(segIndex);
    });

    // Load the blob
    await ws.loadBlob(blob);

    const entry = { ws, blobUrl, duration: ws.getDuration() };
    this.instances.set(segIndex, entry);

    // Evict old instances
    this._evict();

    return entry;
  }

  /**
   * Create a lightweight HTMLAudioElement player without rendering waveform.
   */
  createAudio(segIndex, audioBase64) {
    if (!audioBase64) return null;
    this.destroy(segIndex);

    let blob;
    try {
      blob = this._b64toBlob(audioBase64, 'audio/wav');
    } catch (e) {
      console.warn(`[Audio] base64 decode failed for seg ${segIndex}:`, e);
      return null;
    }

    const blobUrl = URL.createObjectURL(blob);
    const audio = new Audio(blobUrl);
    audio.preload = 'metadata';
    audio.addEventListener('play', () => this._onPlay(segIndex));
    audio.addEventListener('pause', () => this._onPause(segIndex));
    audio.addEventListener('ended', () => this._onFinish(segIndex));
    audio.addEventListener('timeupdate', () => this._onTimeUpdate(segIndex, audio.currentTime || 0, audio.duration || 0));
    audio.addEventListener('loadedmetadata', () => {
      const entry = this.instances.get(segIndex);
      if (entry) entry.duration = Number.isFinite(audio.duration) ? audio.duration : 0;
    });

    const entry = { audio, blobUrl, duration: 0 };
    this.instances.set(segIndex, entry);
    this._evict();
    return entry;
  }

  /**
   * Play a specific segment's audio.
   */
  play(segIndex) {
    const entry = this.instances.get(segIndex);
    if (!entry) return;

    // Pause the currently active one
    if (this.activeIndex !== null && this.activeIndex !== segIndex) {
      const previousIndex = this.activeIndex;
      const current = this.instances.get(previousIndex);
      if (current) {
        try {
          if (current.ws) current.ws.pause();
          if (current.audio) current.audio.pause();
        } catch (e) { /* ignore */ }
      }
      this._highlightSegment(previousIndex, false);
    }

    this.activeIndex = segIndex;
    try {
      if (entry.ws) entry.ws.play();
      else if (entry.audio) entry.audio.play();
    } catch (e) {
      console.warn(`[Audio] Play failed for seg ${segIndex}:`, e);
    }
  }

  /**
   * Pause the currently active segment.
   */
  pause() {
    if (this.activeIndex === null) return;
    const entry = this.instances.get(this.activeIndex);
    if (entry) {
      try {
        if (entry.ws) entry.ws.pause();
        if (entry.audio) entry.audio.pause();
      } catch (e) { /* ignore */ }
    }
    this._highlightSegment(this.activeIndex, false);
    this.activeIndex = null;
    this._updateGlobalUI(false);
  }

  /**
   * Stop and destroy all instances.
   */
  reset() {
    this.pause();
    this._stopDBUpdates();
    for (const [idx, entry] of this.instances) {
      try {
        if (entry.ws) entry.ws.destroy();
        if (entry.audio) {
          entry.audio.pause();
          entry.audio.src = '';
          entry.audio.load();
        }
      } catch (e) { /* ignore */ }
      try { URL.revokeObjectURL(entry.blobUrl); } catch (e) { /* ignore */ }
    }
    this.instances.clear();
    this.activeIndex = null;
    this._updateGlobalUI(false);
  }

  /**
   * Destroy a single instance.
   */
  destroy(segIndex) {
    const entry = this.instances.get(segIndex);
    if (!entry) return;
    if (this.activeIndex === segIndex) {
      this.activeIndex = null;
      this._stopDBUpdates();
    }
    try {
      if (entry.ws) entry.ws.destroy();
      if (entry.audio) {
        entry.audio.pause();
        entry.audio.src = '';
        entry.audio.load();
      }
    } catch (e) { /* ignore */ }
    try { URL.revokeObjectURL(entry.blobUrl); } catch (e) { /* ignore */ }
    this.instances.delete(segIndex);
  }

  /**
   * Get the duration of a segment in seconds.
   */
  getDuration(segIndex) {
    const entry = this.instances.get(segIndex);
    return entry ? entry.duration : 0;
  }

  // ── Private helpers ──────────────────────────────────────

  _b64toBlob(b64, mimeType) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: mimeType });
  }

  _onPlay(segIndex) {
    this.activeIndex = segIndex;
    this._updateGlobalUI(true);
    this._highlightSegment(segIndex, true);
  }

  _onPause(segIndex) {
    this._updateGlobalUI(this.activeIndex !== null && this.activeIndex !== segIndex);
    this._highlightSegment(segIndex, false);
  }

  _onFinish(segIndex) {
    if (this.activeIndex === segIndex) {
      this.activeIndex = null;
      this._updateGlobalUI(false);
      this._highlightSegment(segIndex, false);
    }
  }

  _onTimeUpdate(segIndex, currentTime, duration) {
    if (this.activeIndex !== segIndex) return;
    const m = Math.floor(currentTime / 60);
    const s = Math.floor(currentTime % 60);
    const safeDuration = Number.isFinite(duration) ? duration : 0;
    const dm = Math.floor(safeDuration / 60);
    const ds = Math.floor(safeDuration % 60);
    if (audioTimeDisplay) {
      audioTimeDisplay.textContent =
        `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')} / ${String(dm).padStart(2,'0')}:${String(ds).padStart(2,'0')}`;
    }
  }

  _startDBUpdates(segIndex) {
    this._stopDBUpdates();
    // We don't have real-time PCM access via wavesurfer public API,
    // but we can poll timeupdate and use the segment's SNR + position
    // to simulate dB movement. Real implementation would use AnalyserNode.
    const poll = () => {
      const entry = this.instances.get(segIndex);
      if (!entry || this.activeIndex !== segIndex) {
        this._stopDBUpdates();
        return;
      }
      // Update is handled in _onTimeUpdate
      this.dBUpdateRAF = requestAnimationFrame(poll);
    };
    this.dBUpdateRAF = requestAnimationFrame(poll);
  }

  _stopDBUpdates() {
    if (this.dBUpdateRAF !== null) {
      cancelAnimationFrame(this.dBUpdateRAF);
      this.dBUpdateRAF = null;
    }
  }

  _updateDBMeter(segIndex, currentTime, duration) {
    // Update the dB meter bar for this segment
    const meterFill = document.querySelector(`.db-meter-fill[data-seg="${segIndex}"]`);
    if (!meterFill) return;

    // Simulate dB level based on position in segment
    // (Real implementation would use AnalyserNode from AudioContext)
    const seg = allSegments[segIndex];
    const snr = seg ? (seg.snr_db || 25) : 25;
    // Normalize: higher SNR → quieter noise floor, so dBFS is lower (better)
    // Map: SNR 0→ -30dB, SNR 30→ -5dB
    const baseLevel = -30 + (snr / 30) * 25;
    // Add small variation based on time position
    const variation = Math.sin(currentTime * 8) * 3 + Math.sin(currentTime * 23) * 1.5;
    const dbLevel = Math.max(-60, Math.min(0, baseLevel + variation));

    // Map to 0-100% width (0dB = 100%, -60dB = 0%)
    const pct = ((dbLevel + 60) / 60) * 100;
    meterFill.style.width = Math.max(2, pct) + '%';

    // Color based on level
    if (pct > 70) meterFill.style.background = 'var(--accent-green)';
    else if (pct > 35) meterFill.style.background = 'var(--accent-yellow)';
    else meterFill.style.background = 'var(--accent-red)';
  }

  _updateGlobalUI(active) {
    if (audioPulse) {
      audioPulse.className = 'pulse' + (active ? '' : ' inactive');
    }
    if (audioStatusText) {
      audioStatusText.textContent = active ? '正在播放音频...' : '无音频';
    }
    if (!active && audioTimeDisplay) {
      audioTimeDisplay.textContent = '--:-- / --:--';
    }
  }

  _highlightSegment(segIndex, on) {
    const el = document.querySelector(`.segment[data-seg-index="${segIndex}"]`);
    if (el) {
      if (on) el.classList.add('playing');
      else el.classList.remove('playing');
    }
    const btn = document.querySelector(`.segment-play-btn[data-seg-index="${segIndex}"]`);
    if (btn) {
      if (on) btn.classList.add('playing');
      else btn.classList.remove('playing');
    }
  }

  _evict() {
    if (this.instances.size <= this.maxInstances) return;
    const keys = [...this.instances.keys()].sort((a, b) => a - b);
    const toRemove = keys.slice(0, this.instances.size - this.maxInstances);
    for (const idx of toRemove) {
      if (idx !== this.activeIndex) {
        this.destroy(idx);
      }
    }
  }
}

// Singleton
const audioManager = new AudioPlaybackManager();

// ═══════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════

function formatTime(s) {
  if (s === undefined || s === null) return '--:--';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

function formatDuration(seg) {
  if (!seg) return '--';
  const start = seg.start_sec ?? seg.start;
  const end = seg.end_sec ?? seg.end;
  const startNum = Number(start);
  const endNum = Number(end);
  if (Number.isFinite(startNum) && Number.isFinite(endNum) && endNum >= startNum) {
    return `${(endNum - startNum).toFixed(1)}s`;
  }

  const duration = seg.duration_sec ?? seg.duration;
  if (duration === undefined || duration === null || duration === '') return '--';
  const durationNum = Number(duration);
  if (Number.isFinite(durationNum) && durationNum >= 0) {
    return `${durationNum.toFixed(1)}s`;
  }
  return '--';
}

function formatConfidence(value) {
  if (value === undefined || value === null || value === '') return '--';
  const n = Number(value);
  if (!Number.isFinite(n)) return '--';
  const pct = n <= 1 ? n * 100 : n;
  return `${Math.round(Math.max(0, Math.min(100, pct)))}%`;
}

function confidenceColor(value) {
  if (value === undefined || value === null || value === '') return 'var(--text-muted)';
  const n = Number(value);
  if (!Number.isFinite(n)) return 'var(--text-muted)';
  const pct = n <= 1 ? n * 100 : n;
  if (pct >= 75) return '#3fb950'; // green
  if (pct >= 50) return '#58a6ff'; // blue
  return '#f85149'; // red
}

function confidenceLabel(value) {
  if (value === undefined || value === null || value === '') return '';
  const n = Number(value);
  if (!Number.isFinite(n)) return '';
  const pct = n <= 1 ? n * 100 : n;
  if (pct >= 75) return '高';
  if (pct >= 50) return '中';
  return '低';
}

function showToast(msg) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function updateConnectionStatus(status) {
  if (!connectionStatus) return;
  const map = {
    'connected': '已连接',
    'completed': '已完成',
    'disconnected': '离线',
  };
  connectionStatus.textContent = map[status] || status;
}

// ═══════════════════════════════════════════════════════════
// Segment Rendering
// ═══════════════════════════════════════════════════════════

/**
 * Render a single concise segment card into the transcript body.
 */
function renderSegment(seg, index, autoPlay = false) {
  if (!transcriptBody) return;
  if (emptyState) emptyState.style.display = 'none';

  const displayText = seg.display_text || seg.text || '';
  const durationText = formatDuration(seg);
  const confValue = seg.asr_confidence ?? seg.confidence;
  const confidenceText = formatConfidence(confValue);
  const confColor = confidenceColor(confValue);
  const confLabel = confidenceLabel(confValue);
  const audioBase64 = seg.audio_wav_base64 || seg.audio_base64 || seg.audio;
  const hasAudio = !!audioBase64;

  // Build text with uncertain spans highlighted
  let textHtml = escapeHTML(displayText);
  if (seg.uncertain_spans && seg.uncertain_spans.length > 0) {
    seg.uncertain_spans.forEach(span => {
      const word = span.text || '';
      if (word && textHtml.includes(escapeHTML(word))) {
        textHtml = textHtml.replace(
          escapeHTML(word),
          `<span style="background:rgba(248,81,73,0.15);border-bottom:1px dashed #f85149;border-radius:2px;padding:0 2px;" title="置信度: ${(span.confidence * 100).toFixed(0)}%">${escapeHTML(word)}</span>`
        );
      }
    });
  }

  const card = document.createElement('div');
  card.className = 'segment';
  card.setAttribute('data-seg-index', index);

  card.innerHTML = `
    <div class="segment-top">
      <span class="segment-index">#${index + 1}</span>
      <span class="segment-dur">时长 ${durationText}</span>
      <span class="segment-confidence" style="color:${confColor};font-weight:600;">置信度 ${confidenceText}${confLabel ? ' (' + confLabel + ')' : ''}</span>
      ${hasAudio ? `<button class="segment-play-btn" data-seg-index="${index}" type="button" title="播放/暂停">播放</button>` : ''}
    </div>
    <div class="segment-text">${textHtml}</div>
  `;

  transcriptBody.appendChild(card);

  if (hasAudio) {
    audioManager.createAudio(index, audioBase64);
    const playBtn = card.querySelector(`.segment-play-btn[data-seg-index="${index}"]`);
    if (playBtn) {
      playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (audioManager.activeIndex === index) audioManager.pause();
        else audioManager.play(index);
      });
    }
  }

  transcriptBody.scrollTop = transcriptBody.scrollHeight;
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════
// Sidebar Updates
// ═══════════════════════════════════════════════════════════

function updateStats(segmentsProcessed = null) {
  const segments = allSegments.filter(Boolean);
  const realCount = segmentsProcessed ?? segments.length;
  if (statSegments) statSegments.textContent = realCount;

  const logicCount = segments.reduce((sum, s) => sum + (Array.isArray(s.logic_flags) ? s.logic_flags.length : 0), 0);
  const lowConfCount = segments.reduce((sum, s) => {
    const spans = Array.isArray(s.uncertain_spans) ? s.uncertain_spans.length : 0;
    const uncertainty = s.uncertainty && s.uncertainty.level && s.uncertainty.level !== 'low' ? 1 : 0;
    return sum + spans + uncertainty;
  }, 0);
  const corrCount = segments.reduce((sum, s) => sum + (Array.isArray(s.corrections) ? s.corrections.length : 0), 0);

  if (statLogicFlags) statLogicFlags.textContent = logicCount;
  if (statLowConf) statLowConf.textContent = lowConfCount;
  if (statCorrections) statCorrections.textContent = corrCount;

  const avgConf = segments.length
    ? segments.reduce((sum, s) => sum + Number(s.asr_confidence || 0), 0) / segments.length
    : 0;
  const pct = avgConf > 0 ? Math.round(avgConf * 100) : Math.min(90, 50 + realCount * 3);
  if (overallConfBar) overallConfBar.style.width = pct + '%';
  if (overallConfText) overallConfText.textContent = pct ? pct + '%' : '--';
}

function updateLogicPanel() {
  if (!logicPanel) return;
  const flags = [];
  for (const seg of allSegments) {
    for (const f of (seg.logic_flags || [])) {
      flags.push(f);
    }
  }
  if (flags.length === 0) {
    logicPanel.innerHTML = '<div style="font-size:11px;color:var(--text-muted);text-align:center;padding:10px 0;">暂无逻辑提示</div>';
    return;
  }
  logicPanel.innerHTML = flags.map(f => {
    const icon = f.severity === 'resolved' ? 'OK' : '!';
    return `<div style="font-size:11px;padding:4px 0;border-bottom:1px solid var(--border);">
      <span style="color:${f.severity === 'resolved' ? 'var(--accent-green)' : 'var(--accent-yellow)'};">${icon}</span>
      ${f.message || ''}
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════
// 分析结果面板渲染 (v4.5)
// ═══════════════════════════════════════════════════════════

const SPEAKER_COLORS = ['#4A90D9', '#E8743C', '#50B86C', '#e8b83c', '#8b6ce8', '#e0556a'];

function resetResultsPanel() {
  if (domainCard) domainCard.innerHTML = '<div class="summary-empty">分析中…</div>';
  if (hotwordCard) hotwordCard.innerHTML = '<div class="summary-empty">提取中…</div>';
  if (speakerCard) speakerCard.innerHTML = '<div class="summary-empty">识别中…</div>';
  if (summaryCard) summaryCard.innerHTML = '<div class="summary-empty">会议处理完成后将自动生成摘要…</div>';
}

function renderResultsPanel(data) {
  if (!data) return;

  // 1. 领域
  if (data.domain) {
    renderDomainCard(data.domain);
  } else {
    if (domainCard) domainCard.innerHTML = '<div class="summary-empty">尚未识别</div>';
  }

  // 2. 热词
  renderHotwordCard(data.hotwords || []);

  // 3. 说话人分布
  renderSpeakerCard(data.speaker_stats || {});

  // 4. 摘要
  if (data.summary) {
    renderSummaryCard(data.summary);
  }

  // 5. 自动提取行动项（如果有 segments）
  if (data.segments && data.segments.length > 0) {
    fetchAndRenderActionItems(data.segments);
  } else if (allSegments.length > 0) {
    fetchAndRenderActionItems(allSegments);
  }
}

function renderDomainCard(domain) {
  const card = document.getElementById('domainCard');
  if (!card) return;

  if (!domain || !domain.domain) {
    card.innerHTML = '<div class="summary-empty">未检测到领域</div>';
    return;
  }

  const pct = domain.confidence ? Math.round(domain.confidence * 100) : '--';
  const methodLabel = domain.method === 'llm' ? 'LLM 推理' : (domain.method === 'demo' ? '演示' : '规则匹配');

  let h = `<div class="domain-badge">${escapeHTML(domain.domain)}</div>`;
  h += `<div class="domain-meta">`;
  h += `<span>置信度 ${pct}%</span>`;
  h += `<span class="dot"></span>`;
  h += `<span>${methodLabel}</span>`;
  h += `</div>`;

  if (domain.sub_domains && domain.sub_domains.length > 0) {
    h += '<div class="domain-sub">';
    domain.sub_domains.forEach(sd => { h += `<span class="domain-sub-tag">${escapeHTML(sd)}</span>`; });
    h += '</div>';
  }

  if (domain.matched_terms && domain.matched_terms.length > 0) {
    h += '<div class="domain-matched">匹配词：' + domain.matched_terms.slice(0, 6).map(t => escapeHTML(t)).join(' · ') + '</div>';
  }

  if (domain.reason) {
    h += `<div class="domain-matched" style="margin-top:2px;">${escapeHTML(domain.reason)}</div>`;
  }

  card.innerHTML = h;
}

function renderHotwordCard(hotwords) {
  const card = document.getElementById('hotwordCard');
  if (!card) return;

  if (!hotwords || hotwords.length === 0) {
    card.innerHTML = '<div class="summary-empty">暂未提取到关键术语</div>';
    return;
  }

  const top = hotwords.slice(0, 20);
  let h = '';
  top.forEach((hw, i) => {
    const cls = i < 5 ? 'hotword-tag top' : 'hotword-tag';
    const label = typeof hw === 'string' ? hw : (hw.word || hw);
    h += `<span class="${cls}">${escapeHTML(label)}</span>`;
  });
  if (hotwords.length > 20) {
    h += `<span class="hotword-tag" style="opacity:0.4;">+${hotwords.length - 20}</span>`;
  }

  card.innerHTML = h;
}

function renderSpeakerCard(stats) {
  const card = document.getElementById('speakerCard');
  if (!card) return;

  if (!stats || Object.keys(stats).length === 0) {
    card.innerHTML = '<div class="summary-empty">暂未识别说话人</div>';
    return;
  }

  const entries = Object.entries(stats).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, e) => s + e[1], 0);

  let h = '';
  entries.forEach(([name, count], i) => {
    const pct = Math.round(count / total * 100);
    const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
    h += `<div class="speaker-row">
      <div class="speaker-dot" style="background:${color};"></div>
      <div class="speaker-name" title="${escapeHTML(name)}">${escapeHTML(name)}</div>
      <div class="speaker-bars"><div class="speaker-bar-bg"><div class="speaker-bar-fill" style="width:${pct}%;background:${color};"></div></div></div>
      <div class="speaker-count">${count} 段 (${pct}%)</div>
    </div>`;
  });

  card.innerHTML = h;
}

function renderSummaryCard(summary) {
  const card = document.getElementById('summaryCard');
  if (!card) return;

  const hasContent = (summary.summary && summary.summary.trim()) ||
    (summary.topics && summary.topics.length > 0) ||
    (summary.action_items && summary.action_items.length > 0);

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

  if (summary.action_items && summary.action_items.length > 0) {
    h += '<div class="summary-section"><div class="summary-section-title">行动项 / TODO</div>';
    summary.action_items.forEach((a, i) => {
      const item = typeof a === 'string' ? { task: a } : a;
      const task = item.task || '';
      const assignee = item.assignee || '';
      const deadline = item.deadline || '';
      const priority = item.priority || 'medium';
      const source = item.source_text || '';
      const speaker = item.speaker || '';

      // Priority color
      const pColor = priority === 'high' ? '#f85149' : priority === 'low' ? '#8b949e' : '#58a6ff';
      const pLabel = priority === 'high' ? '高' : priority === 'low' ? '低' : '中';

      h += `<div style="margin:6px 0;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;border-left:3px solid ${pColor};">`;
      // Row 1: task + priority badge
      h += `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">`;
      h += `<span style="font-size:11px;padding:2px 6px;border-radius:3px;background:${pColor}33;color:${pColor};font-weight:600;">${pLabel}</span>`;
      h += `<span style="font-size:13px;color:#e6edf3;font-weight:500;">${escapeHTML(task)}</span>`;
      h += `</div>`;
      // Row 2: meta (assignee / deadline / speaker)
      const metaParts = [];
      if (assignee) metaParts.push(`👤 ${escapeHTML(assignee)}`);
      if (deadline) metaParts.push(`⏰ ${escapeHTML(deadline)}`);
      if (speaker) metaParts.push(`🎙 ${escapeHTML(speaker)}`);
      if (metaParts.length) {
        h += `<div style="margin-top:4px;font-size:11px;color:#8b949e;display:flex;gap:12px;flex-wrap:wrap;">${metaParts.join('')}</div>`;
      }
      // Row 3: source text
      if (source) {
        h += `<div style="margin-top:6px;font-size:11px;color:#6e7681;border-top:1px solid #21262d;padding-top:4px;font-style:italic;">"${escapeHTML(source)}"</div>`;
      }
      h += `</div>`;
    });
    h += '</div>';
  }

  card.innerHTML = h;
}

// ═══════════════════════════════════════════════════════════
// Action Items: fetch from API and render
// ═══════════════════════════════════════════════════════════

async function fetchAndRenderActionItems(segments) {
  if (!segments || segments.length === 0) return;
  const card = document.getElementById('summaryCard');
  if (!card) return;

  try {
    const resp = await fetch(`${API_BASE}/api/action-items`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments: segments.map(s => ({
        speaker: s.speaker || '',
        text: s.text || s.raw_text || '',
        start: s.start || s.timestamp || 0,
        end: s.end || 0,
      })) }),
    });
    const data = await resp.json();
    if (data.action_items && data.action_items.length > 0) {
      // Append action items to summary card
      let h = card.innerHTML;
      h += '<div class="summary-section"><div class="summary-section-title">行动项 / TODO (自动提取)</div>';
      data.action_items.forEach((a, i) => {
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
      h += '</div>';
      card.innerHTML = h;
    }
  } catch (e) {
    console.error('[DiTing] Action items fetch failed:', e);
  }
}

// ═══════════════════════════════════════════════════════════
// Demo Mode
// ═══════════════════════════════════════════════════════════

async function startDemo() {
  if (isPlaying) return;
  resetAll();

  try {
    const resp = await fetch(`${API_BASE}/api/meeting/demo`);
    demoData = await resp.json();
  } catch (e) {
    console.log('Using embedded demo data');
    demoData = getEmbeddedDemoData();
  }

  if (meetingTitleDisplay) {
    meetingTitleDisplay.textContent = demoData.title || '演示会议';
  }
  allSegments = demoData.segments || [];

  isPlaying = true;
  const btnStart = $('btnStart');
  if (btnStart) { btnStart.textContent = '演示中...'; btnStart.disabled = true; }
  updateConnectionStatus('connected');

  currentSegmentIndex = -1;
  playNextDemoSegment();
}

function playNextDemoSegment() {
  if (!isPlaying) return;
  currentSegmentIndex++;
  if (currentSegmentIndex >= allSegments.length) {
    finishDemo();
    return;
  }

  const seg = allSegments[currentSegmentIndex];
  renderSegment(seg, currentSegmentIndex, false);  // demo: no audio, no auto-play
  updateStats(currentSegmentIndex + 1);
  updateProgress(currentSegmentIndex);
  if (seg.logic_flags && seg.logic_flags.length > 0) {
    updateLogicPanel();
    for (const f of seg.logic_flags) {
      if (f.severity === 'warning') showToast('逻辑提示：' + (f.message || '').substring(0, 60));
    }
  }

  const delay = 1500;
  playbackTimer = setTimeout(playNextDemoSegment, delay);
}

function finishDemo() {
  isPlaying = false;
  const btnStart = $('btnStart');
  if (btnStart) { btnStart.textContent = '演示'; btnStart.disabled = false; }
  updateConnectionStatus('completed');
  showToast('演示完成');
}

function updateProgress(segIndex) {
  const pct = ((segIndex + 1) / Math.max(1, allSegments.length)) * 100;
  if (progressFill) progressFill.style.width = pct + '%';
  if (segmentCounter) segmentCounter.textContent = `${segIndex + 1}/${allSegments.length}`;
  const last = allSegments[segIndex];
  if (timeDisplay) timeDisplay.textContent = formatTime(last ? (last.end_sec || last.end || 0) : 0);
}

function getEmbeddedDemoData() {
  return {
    "title": "Q3 产品复盘演示",
    "date": "2024-07-14 14:00",
    "duration": "04:30",
    "participants": [],
    "hotwords": ["BERT", "Q3", "Conversion"],
    "segments": [
      {"start":0,"end":5,"snr_db":28,"quality_label":"high","display_text":"演示片段 1：这是模拟会议数据，暂无音频。"},
      {"start":6,"end":12,"snr_db":26,"quality_label":"high","display_text":"演示片段 2：上传真实音频后可以查看实时波形。"},
      {"start":13,"end":18,"snr_db":27,"quality_label":"high","display_text":"演示片段 3：可查看后端日志了解处理细节。"},
    ],
    "summary": {"topics":["Demo"], "todos":[], "low_confidence_spots":[], "stats":{}}
  };
}

// ═══════════════════════════════════════════════════════════
// File Upload (with real-time WebSocket + audio waveforms)
// ═══════════════════════════════════════════════════════════

function saveUploadTaskRef(fileId, filename) {
  try {
    localStorage.setItem(UPLOAD_TASK_STORAGE_KEY, JSON.stringify({ fileId, filename, savedAt: Date.now() }));
  } catch (e) { /* ignore */ }
}

function clearUploadTaskRef() {
  try { localStorage.removeItem(UPLOAD_TASK_STORAGE_KEY); } catch (e) { /* ignore */ }
}

async function fetchUploadStatus(fileId) {
  const resp = await fetch(`${API_BASE}/api/audio/upload/${fileId}/status`);
  return resp.json();
}

function renderUploadSnapshot(task) {
  if (!task || !task.file_id) return;
  uploadFileId = task.file_id;
  if (task.filename && meetingTitleDisplay) meetingTitleDisplay.textContent = task.filename;

  if (task.segments && task.segments.length > 0) {
    audioManager.reset();
    allSegments = task.segments;
    if (transcriptBody) transcriptBody.innerHTML = '';
    allSegments.forEach((seg, i) => renderSegment(seg, i, false));
    if (emptyState) emptyState.style.display = 'none';
    if (segmentCounter) segmentCounter.textContent = `${allSegments.length}/${task.total_estimated || allSegments.length}`;
    updateStats(allSegments.length);
    updateLogicPanel();
  }

  if (processingPanel && task.status && !['completed', 'demo_mode'].includes(task.status)) {
    processingPanel.innerHTML = `<div class="upload-status-card">
      <div class="stage">阶段：${escapeHTML(task.progress_stage || task.status)}</div>
      <div style="color:var(--text-muted);">${Math.round((task.progress_fraction || 0) * 100)}% · ${escapeHTML(task.filename || '')}</div>
    </div>`;
  }

  if (task.status === 'completed' || task.status === 'demo_mode') {
    renderResultsPanel(task);
    finishUploadUI('complete', { keepTaskRef: true });
  } else if (task.status === 'error') {
    showToast('上传任务失败：' + (task.error || '未知错误'));
    finishUploadUI('error');
  } else if (task.status === 'cancelled') {
    showToast('上传任务已取消');
    finishUploadUI('cancelled');
  } else if (task.status === 'processing' || task.status === 'cancelling') {
    updateConnectionStatus('connected');
  }
}

async function restoreUploadTaskOnLoad() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(UPLOAD_TASK_STORAGE_KEY) || 'null'); } catch (e) { saved = null; }
  if (!saved?.fileId) return;
  try {
    const status = await fetchUploadStatus(saved.fileId);
    if (!status.ok || !status.task) {
      clearUploadTaskRef();
      return;
    }
    renderUploadSnapshot(status.task);
    if (status.task.status === 'processing' || status.task.status === 'cancelling') {
      uploadWs = new WebSocket(`ws://localhost:8765/ws/upload/${saved.fileId}`);
      uploadWs.onmessage = (event) => {
        try { handleUploadWSMessage(JSON.parse(event.data), resetGeneration); }
        catch (e) { console.warn('[Upload] Bad restored WS message:', e); }
      };
    }
  } catch (e) {
    console.warn('[Upload] restore task failed:', e);
  }
}

/**
 * Render segments from HTTP response (fallback when WS doesn't deliver).
 */
function handleUploadHTTPResponse(result, generation = resetGeneration) {
  if (generation !== resetGeneration) return;
  if (!result || uploadWsDelivered) return;

  if (result.status === 'completed' && result.segments && result.segments.length > 0) {
    clearUploadTimers();
    console.log('[Upload] Rendering from HTTP response:', result.segments.length, 'segments');
    uploadWsDelivered = true;  // prevent double processing
    allSegments = result.segments;
    if (transcriptBody) transcriptBody.innerHTML = '';
    result.segments.forEach((seg, i) => renderSegment(seg, i, false));
    if (segmentCounter) segmentCounter.textContent = `${result.segments.length}/${result.segments.length}`;
    updateStats(result.segments_count || result.segments.length);
    updateLogicPanel();
    renderResultsPanel(result);
    showToast(`完成：${result.segments_count || result.segments.length} 个片段`);
    finishUploadUI('complete');
  } else if (result.status === 'demo_mode') {
    uploadWsDelivered = true;  // prevent double processing
    showToast('X-ASR 未加载，使用演示数据');
    if (result.demo_data) {
      demoData = result.demo_data;
      startDemo();
    }
    finishUploadUI('demo');
  } else if (result.status === 'processing') {
    // Server is processing — wait for WS or fallback timer
    console.log('[Upload] Server processing, waiting for WS...');
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const generation = resetGeneration;
  const fileId = generateUUID();
  saveUploadTaskRef(fileId, file.name);
  uploadFileId = fileId;
  uploadHttpResult = null;
  uploadWsDelivered = false;
  const btnUpload = $('btnUpload');
  if (btnUpload) { btnUpload.textContent = '连接中...'; btnUpload.disabled = true; }

  if (uploadWs) { try { uploadWs.close(); } catch(e) {} uploadWs = null; }
  uploadAbortController = new AbortController();

  showToast(`正在上传：${file.name}`);
  if (meetingTitleDisplay) meetingTitleDisplay.textContent = file.name;

  // Clear previous state without cancelling this new upload generation
  resetAll({ cancelTasks: false });
  allSegments = [];

  // Show uploading state
  if (transcriptBody) transcriptBody.innerHTML = '';
  if (emptyState) {
    emptyState.style.display = 'flex';
    const msgEl = emptyState.querySelector('div:nth-child(2)');
    if (msgEl) msgEl.textContent = `处理中：${file.name}`;
  }
  if (processingPanel) {
    processingPanel.innerHTML = `<div class="upload-status-card">
      <div class="stage">正在上传：${file.name}</div>
      <div style="color:var(--text-muted);">${(file.size/1024/1024).toFixed(1)} MB</div>
    </div>`;
  }

  // ── Fallback timer: if WS doesn't deliver within 30s, use HTTP result ──
  uploadFallbackTimer = setTimeout(() => {
    if (generation !== resetGeneration) return;
    if (!uploadWsDelivered && uploadHttpResult) {
      console.log('[Upload] WS timeout — rendering from HTTP fallback');
      handleUploadHTTPResponse(uploadHttpResult, generation);
    } else if (!uploadWsDelivered && !uploadHttpResult) {
      showToast('处理超时，请检查后端日志');
      finishUploadUI('timeout');
    }
  }, 30000);

  try {
    uploadWs = new WebSocket(`ws://localhost:8765/ws/upload/${fileId}`);

    uploadWs.onopen = async () => {
      console.log('[Upload] WS connected');
      if (btnUpload) btnUpload.textContent = '上传中...';

      // Now upload the file via HTTP
      try {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch(`${API_BASE}/api/audio/upload?file_id=${fileId}`, {
          method: 'POST', body: formData, signal: uploadAbortController?.signal,
        });
        const result = await resp.json();
        console.log('[Upload] HTTP status:', result.status);
        uploadHttpResult = result;

        if (generation !== resetGeneration) return;
        if (result.status === 'error') {
          showToast('错误：' + (result.error || '未知错误'));
          clearUploadTimers();
          finishUploadUI('error');
        } else if (result.status === 'demo_mode') {
          clearUploadTimers();
          handleUploadHTTPResponse(result, generation);
        } else if (result.status === 'completed') {
          handleUploadHTTPResponse(result, generation);
        }
        // For 'processing': wait for WS to deliver segments
      } catch (e) {
        if (generation !== resetGeneration || e.name === 'AbortError') return;
        console.error('[Upload] Fetch error:', e);
        if (!uploadWsDelivered) {
          clearUploadTimers();
          uploadFallback(file, generation);
        }
      }
    };

    uploadWs.onmessage = (event) => {
      try {
        if (generation !== resetGeneration) return;
        const msg = JSON.parse(event.data);
        handleUploadWSMessage(msg, generation);
      } catch (e) {
        console.warn('[Upload] Bad WS message:', e);
      }
    };

    uploadWs.onerror = () => {
      console.error('[Upload] WS error');
    };

    uploadWs.onclose = () => {
      console.log('[Upload] WS closed');
      // If WS closed without delivering segments, use HTTP fallback
      if (!uploadWsDelivered && uploadHttpResult) {
        if (uploadFallbackTimer !== null) {
          clearTimeout(uploadFallbackTimer);
          uploadFallbackTimer = null;
        }
        // Small delay to let any in-flight messages arrive
        uploadCloseFallbackTimer = setTimeout(() => {
          if (generation !== resetGeneration) return;
          if (!uploadWsDelivered) {
            handleUploadHTTPResponse(uploadHttpResult, generation);
          }
        }, 1500);
      }
    };

  } catch (e) {
    console.error('[Upload] WS creation failed:', e);
    clearUploadTimers();
    uploadFallback(file, generation);
  }
}

function clearUploadTimers() {
  if (uploadFallbackTimer !== null) {
    clearTimeout(uploadFallbackTimer);
    uploadFallbackTimer = null;
  }
  if (uploadCloseFallbackTimer !== null) {
    clearTimeout(uploadCloseFallbackTimer);
    uploadCloseFallbackTimer = null;
  }
}

async function cancelUploadTask() {
  const fileId = uploadFileId;
  clearUploadTimers();
  showToast('正在取消上传任务...');

  if (fileId) {
    try {
      await fetch(`${API_BASE}/api/audio/upload/${fileId}/cancel`, { method: 'POST' });
    } catch (e) {
      console.warn('[Upload] cancel request failed:', e);
    }
  }

  if (uploadAbortController) {
    try { uploadAbortController.abort(); } catch (e) { /* ignore */ }
    uploadAbortController = null;
  }
  if (uploadWs) {
    try { uploadWs.close(); } catch (e) { /* ignore */ }
    uploadWs = null;
  }
  uploadHttpResult = null;
  uploadWsDelivered = false;
  uploadFileId = null;
  clearUploadTaskRef();
  const btnUpload = $('btnUpload');
  if (btnUpload) { btnUpload.textContent = '上传'; btnUpload.disabled = false; }
  const fileInput = $('fileUpload');
  if (fileInput) fileInput.value = '';
}

async function uploadFallback(file, generation = resetGeneration) {
  if (generation !== resetGeneration) return;
  // Already tried — don't double-upload
  if (uploadHttpResult) {
    handleUploadHTTPResponse(uploadHttpResult, generation);
    return;
  }

  showToast('正在上传，暂无实时视图...');
  const fileId = uploadFileId || generateUUID();
  try {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(`${API_BASE}/api/audio/upload?file_id=${fileId}`, {
      method: 'POST', body: formData,
    });
    const result = await resp.json();
    uploadHttpResult = result;
    handleUploadHTTPResponse(result, generation);
  } catch (e) {
    console.error('[Upload] Fallback failed:', e);
    showToast('上传失败，请检查后端是否运行');
    finishUploadUI('error');
  }
  const btnUpload = $('btnUpload');
  if (btnUpload) { btnUpload.textContent = '上传'; btnUpload.disabled = false; }
  const fileInput = $('fileUpload');
  if (fileInput) fileInput.value = '';
}

function handleUploadWSMessage(msg, generation = resetGeneration) {
  if (generation !== resetGeneration) return;
  const panel = processingPanel;

  switch (msg.type) {
    case 'connected':
      console.log('[Upload] WS ready, waiting for file...');
      break;

    case 'snapshot':
      renderUploadSnapshot(msg.data?.task);
      break;

    case 'status':
      if (panel) {
        panel.innerHTML = `<div class="upload-status-card">
          <div class="stage">处理中：${escapeHTML(msg.data.filename || '')}</div>
          <div style="color:var(--text-muted);">引擎：${escapeHTML(msg.data.engine || '')}</div>
        </div>`;
      }
      break;

    case 'progress':
      if (panel) {
        const progEl = panel.querySelector('.stage');
        if (progEl) progEl.textContent = `阶段：${msg.data.stage} (${(msg.data.fraction*100).toFixed(0)}%)`;
      }
      break;

    case 'segment': {
      uploadWsDelivered = true;
      const seg = msg.data.segment;
      const segIndex = msg.data.segment_index;

      // Store in allSegments (preserve existing if already set)
      if (!allSegments[segIndex]) {
        allSegments[segIndex] = seg;
      }

      // Do not auto-play recognized audio; users can click a segment to play it.
      const autoPlay = false;

      // Check if segment card already exists
      const existingCard = document.querySelector(`.segment[data-seg-index="${segIndex}"]`);
      if (!existingCard) {
        renderSegment(seg, segIndex, autoPlay);
      }

      // Update counters
      const realCount = allSegments.filter(Boolean).length;
      const totalEst = msg.data.total_estimated || realCount || 50;
      if (segmentCounter) segmentCounter.textContent = `${realCount}/${totalEst}`;
      updateStats(realCount);
      if (progressFill) progressFill.style.width = Math.min(95, (realCount / Math.max(1, totalEst)) * 100) + '%';

      updateLogicPanel();
      break;
    }

    case 'complete':
      uploadWsDelivered = true;
      clearUploadTimers();

      if (msg.data?.status === 'demo_mode') {
        showToast('X-ASR 未加载，使用演示数据');
        if (msg.data.demo_data) {
          demoData = msg.data.demo_data;
          startDemo();
        }
        finishUploadUI('demo');
        break;
      }

      showToast(`完成：${msg.data.segments_count || allSegments.filter(Boolean).length} 个片段`);
      // Use the final complete payload as the source of truth so audio and analysis fields are present.
      if (msg.data.segments && msg.data.segments.length > 0) {
        audioManager.reset();
        allSegments = msg.data.segments;
        if (transcriptBody) transcriptBody.innerHTML = '';
        allSegments.forEach((seg, i) => renderSegment(seg, i, false));
        if (segmentCounter) segmentCounter.textContent = `${allSegments.length}/${allSegments.length}`;
      }
      updateStats(allSegments.filter(Boolean).length);
      updateLogicPanel();
      renderResultsPanel(msg.data);
      finishUploadUI('complete');
      break;

    case 'error':
      showToast('错误：' + (msg.data?.message || '未知错误'));
      // Try HTTP fallback if WS delivered nothing
      if (!uploadWsDelivered && uploadHttpResult) {
        handleUploadHTTPResponse(uploadHttpResult, generation);
      } else {
        finishUploadUI('error');
      }
      break;

    case 'cancelled':
      showToast('上传处理已取消');
      finishUploadUI('cancelled');
      break;

    case 'timeout':
      showToast('处理超时');
      if (!uploadWsDelivered && uploadHttpResult) {
        handleUploadHTTPResponse(uploadHttpResult, generation);
      } else {
        finishUploadUI('timeout');
      }
      break;
  }
}

function finishUploadUI(status, options = {}) {
  const { keepTaskRef = false } = options;
  clearUploadTimers();
  const btnUpload = $('btnUpload');
  if (btnUpload) { btnUpload.textContent = '上传'; btnUpload.disabled = false; }

  if (status === 'complete') {
    updateConnectionStatus('completed');
    if (progressFill) progressFill.style.width = '100%';
    if (overallConfBar) { overallConfBar.style.width = '85%'; overallConfBar.style.background = 'var(--accent-green)'; }
    if (overallConfText) overallConfText.textContent = '85%';
  } else if (status === 'cancelled') {
    updateConnectionStatus('disconnected');
    clearUploadTaskRef();
  } else {
    updateConnectionStatus('disconnected');
    if (!keepTaskRef && status !== 'timeout') clearUploadTaskRef();
  }

  if (uploadWs) { uploadWs.close(); uploadWs = null; }
  const fileInput = $('fileUpload');
  if (fileInput) fileInput.value = '';
}

// ═══════════════════════════════════════════════════════════
// Live Mic Recording
// ═══════════════════════════════════════════════════════════

function setMicOrbVisible(visible, subtitle = '') {
  if (!micOrbPanel) return;
  micOrbPanel.classList.toggle('visible', visible);
  micOrbPanel.classList.toggle('listening', visible && isRecording);
  micOrbPanel.setAttribute('aria-hidden', visible ? 'false' : 'true');
  if (micOrbTitle) micOrbTitle.textContent = visible ? '正在听你说话' : '麦克风已停止';
  if (micOrbSubtitle) micOrbSubtitle.textContent = subtitle || (visible ? '音量越大，语音球越活跃' : '');
  if (!visible) updateMicOrbLevel(0);
}

function updateMicOrbLevel(level) {
  const safeLevel = Math.max(0, Math.min(1, level || 0));
  const scale = 1 + safeLevel * 0.42;
  if (micOrbWrap) {
    micOrbWrap.style.setProperty('--orb-level', safeLevel.toFixed(3));
    micOrbWrap.style.setProperty('--orb-scale', scale.toFixed(3));
    micOrbWrap.style.setProperty('--orb-ring-opacity', (0.25 + safeLevel * 0.65).toFixed(3));
    micOrbWrap.style.setProperty('--orb-shadow', (18 + safeLevel * 34).toFixed(1) + 'px');
    micOrbWrap.style.setProperty('--orb-blur', (0.5 + safeLevel * 3).toFixed(1) + 'px');
    micOrbWrap.style.setProperty('--orb-glow-opacity', (0.82 + safeLevel * 0.18).toFixed(3));
    micOrbWrap.style.setProperty('--orb-core-scale', (1 + safeLevel * 0.22).toFixed(3));
    micOrbWrap.style.setProperty('--orb-core-shadow', (20 + safeLevel * 50).toFixed(1) + 'px');
  }
  if (micOrbWave) {
    const bars = micOrbWave.querySelectorAll('span');
    bars.forEach((bar, i) => {
      const phase = Math.sin(performance.now() / 120 + i * 0.72) * 0.5 + 0.5;
      const barLevel = Math.max(0.06, safeLevel * (0.42 + phase * 0.88));
      bar.style.setProperty('--bar-level', barLevel.toFixed(3));
      bar.style.setProperty('--bar-height', (5 + barLevel * 24).toFixed(1) + 'px');
      bar.style.setProperty('--bar-scale', (0.45 + barLevel).toFixed(3));
    });
  }
}

function startMicVisualizer() {
  stopMicVisualizer();
  if (!micAnalyser) return;

  const data = new Float32Array(micAnalyser.fftSize);
  const tick = () => {
    if (!isRecording || !micAnalyser) {
      stopMicVisualizer();
      return;
    }
    micAnalyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const rms = Math.sqrt(sum / data.length);
    const level = Math.min(1, rms * 9);
    micLevelSmoothed = micLevelSmoothed * 0.72 + level * 0.28;
    updateMicOrbLevel(micLevelSmoothed);
    micVisualizerRAF = requestAnimationFrame(tick);
  };
  tick();
}

function stopMicVisualizer() {
  if (micVisualizerRAF !== null) {
    cancelAnimationFrame(micVisualizerRAF);
    micVisualizerRAF = null;
  }
  micLevelSmoothed = 0;
  updateMicOrbLevel(0);
}

async function toggleMic() {
  const btnMic = $('btnMic');
  const micStatusEl = micStatus;

  if (isRecording) {
    stopMicRecording();
    if (btnMic) { btnMic.textContent = '麦克风'; btnMic.classList.remove('active'); }
    if (micStatusEl) micStatusEl.textContent = '';
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
    });
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    // Resume AudioContext (browsers start it suspended)
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    micAudioContext = audioContext;
    const source = audioContext.createMediaStreamSource(micStream);
    micAnalyser = audioContext.createAnalyser();
    micAnalyser.fftSize = 1024;
    micAnalyser.smoothingTimeConstant = 0.78;
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    try {
      micWs = new WebSocket('ws://localhost:8765/ws/live');
      micWs.onopen = () => {
        console.log('[Mic] WebSocket connected to /ws/live');
        if (micStatusEl) { micStatusEl.textContent = 'X-ASR Live'; micStatusEl.style.color = 'var(--accent-green)'; }
        setMicOrbVisible(true, '已连接实时识别，开始说话');
      };
      micWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log('[Mic] WS message:', msg.type, msg.data?.text?.slice(0, 40) || '');
        if (msg.type === 'live_result' && msg.data.text) {
          if (msg.data.is_partial) {
            setMicOrbVisible(true, '正在实时转写...');
          } else if (msg.data.is_final) {
            setMicOrbVisible(true, '已收到一句完整识别结果');
          }
          updateMicResult(msg.data);
        }
      };
      micWs.onerror = (e) => {
        console.error('[Mic] WebSocket error:', e);
        if (micStatusEl) { micStatusEl.textContent = 'WS Error'; micStatusEl.style.color = 'var(--accent-red)'; }
        setMicOrbVisible(true, '实时识别连接失败，但仍在显示麦克风输入');
        showToast('WebSocket 连接失败，请检查后端是否运行');
      };
      micWs.onclose = () => {
        console.log('[Mic] WebSocket closed');
      };
    } catch (e) {
      console.error('[Mic] WebSocket creation failed:', e);
      if (micStatusEl) { micStatusEl.textContent = '未连接 WS'; micStatusEl.style.color = 'var(--accent-yellow)'; }
      setMicOrbVisible(true, '未连接后端，仅显示麦克风音量');
    }

    let chunkCount = 0;
    const actualSampleRate = audioContext.sampleRate;
    console.log('[Mic] AudioContext sampleRate:', actualSampleRate, '(requested 16000)');

    processor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      chunkCount++;

      // Log audio level every 10 chunks
      if (chunkCount % 10 === 0) {
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
        const rms = Math.sqrt(sum / inputData.length);
        console.log(`[Mic] chunk #${chunkCount}, samples=${inputData.length}, RMS=${rms.toFixed(4)}, sr=${actualSampleRate}`);
      }

      if (micWs && micWs.readyState === WebSocket.OPEN) {
        // Resample to 16kHz if AudioContext is running at a different rate
        let audio16k = inputData;
        if (actualSampleRate !== 16000) {
          const ratio = 16000 / actualSampleRate;
          const newLen = Math.round(inputData.length * ratio);
          audio16k = new Float32Array(newLen);
          for (let i = 0; i < newLen; i++) {
            const srcIdx = i / ratio;
            const idx0 = Math.floor(srcIdx);
            const idx1 = Math.min(idx0 + 1, inputData.length - 1);
            const frac = srcIdx - idx0;
            audio16k[i] = inputData[idx0] * (1 - frac) + inputData[idx1] * frac;
          }
        }

        const bytes = new Uint8Array(audio16k.buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
        micWs.send(JSON.stringify({ action: 'process_chunk', audio: btoa(binary) }));
      }
    };

    // Connect through zero-gain node to prevent audio feedback to speakers
    const silentGain = audioContext.createGain();
    silentGain.gain.value = 0;
    source.connect(micAnalyser);
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);
    isRecording = true;
    setMicOrbVisible(true);
    startMicVisualizer();
    if (btnMic) { btnMic.textContent = '停止'; btnMic.classList.add('active'); }
    if (micStatusEl) { micStatusEl.textContent = '录音中...'; micStatusEl.style.color = 'var(--accent-red)'; }
    showToast('录音已开始');
  } catch (err) {
    console.error('Mic error:', err);
    stopMicVisualizer();
    setMicOrbVisible(false);
    if (micStatusEl) { micStatusEl.textContent = '麦克风被拒绝'; micStatusEl.style.color = 'var(--accent-red)'; }
    showToast('无法访问麦克风');
  }
}

function stopMicRecording() {
  isRecording = false;
  stopMicVisualizer();
  setMicOrbVisible(false);
  micAnalyser = null;
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (micWs) {
    if (micWs.readyState === WebSocket.OPEN) {
      micWs.send(JSON.stringify({ action: 'stop' }));
      // Give backend 800ms to process queued chunks and send final results
      setTimeout(() => {
        if (micWs) { micWs.close(); micWs = null; }
      }, 800);
    } else {
      micWs.close();
      micWs = null;
    }
  }
  if (micAudioContext) {
    micAudioContext.close().catch(() => {});
    micAudioContext = null;
  }
  // Auto-extract action items after mic recording stops
  if (allSegments.length > 0) {
    setTimeout(() => fetchAndRenderActionItems(allSegments), 1200);
  }
}

function updateMicResult(data) {
  if (!transcriptBody) return;
  if (emptyState) emptyState.style.display = 'none';

  const liveEl = document.getElementById('live-result');
  if (data.is_partial) {
    const confValue = data.asr_confidence ?? data.confidence;
    const confidenceText = formatConfidence(confValue);
    const confColor = confidenceColor(confValue);
    if (liveEl) {
      const textEl = liveEl.querySelector('.segment-text');
      const confidenceEl = liveEl.querySelector('.segment-confidence');
      if (textEl) textEl.textContent = data.text || '...';
      if (confidenceEl) {
        confidenceEl.textContent = `置信度 ${confidenceText}`;
        confidenceEl.style.color = confColor;
        confidenceEl.style.fontWeight = '600';
      }
      return;
    }
    const html = `<div class="segment" id="live-result" style="opacity:0.7;">
      <div class="segment-top">
        <span class="segment-index">实时</span>
        <span class="segment-dur">时长 ${formatDuration(data)}</span>
        <span class="segment-confidence" style="color:${confColor};font-weight:600;">置信度 ${confidenceText}</span>
      </div>
      <div class="segment-text">${escapeHTML(data.text || '（监听中...）')}</div>
    </div>`;
    transcriptBody.insertAdjacentHTML('beforeend', html);
  } else if (data.is_final && data.text) {
    if (liveEl) liveEl.remove();
    const idx = allSegments.length;
    allSegments.push(data);
    renderSegment(data, idx, false);
    updateStats(allSegments.length);
  }
  transcriptBody.scrollTop = transcriptBody.scrollHeight;
}

// ═══════════════════════════════════════════════════════════
// Debug Log Panel
// ═══════════════════════════════════════════════════════════

function toggleLogPanel() {
  const panel = $('logPanel');
  logVisible = !logVisible;
  if (logVisible) {
    panel.classList.add('visible');
  } else {
    panel.classList.remove('visible');
  }
}

function connectLogStream() {
  try {
    logWs = new WebSocket('ws://localhost:8765/ws/logs');
    logWs.onmessage = (event) => {
      if (!logVisible) return;
      const msg = JSON.parse(event.data);
      if (msg.type === 'logs') {
        const body = $('logPanelBody');
        if (body) {
          body.innerHTML = msg.data.slice(-30).map(l =>
            `<div style="color:${l.level==='ERROR'?'var(--accent-red)':l.level==='WARNING'?'var(--accent-yellow)':'var(--text-muted)'};font-size:10px;">
              [${(l.timestamp||'').substring(11,19)}] ${l.level} ${(l.message||'').substring(0,120)}
            </div>`
          ).join('');
          body.scrollTop = body.scrollHeight;
        }
      }
    };
    logWs.onerror = () => { /* silent */ };
  } catch (e) { /* silent */ }
}

// ═══════════════════════════════════════════════════════════
// Reset
// ═══════════════════════════════════════════════════════════

function resetAll(options = {}) {
  const { cancelTasks = true } = options;

  isPlaying = false;
  currentSegmentIndex = -1;
  clearTimeout(playbackTimer);
  if (cancelTasks) {
    const fileIdToCancel = uploadFileId;
    if (fileIdToCancel) {
      fetch(`${API_BASE}/api/audio/upload/${fileIdToCancel}/cancel`, { method: 'POST' })
        .catch(e => console.warn('[Upload] cancel request failed:', e));
    }
    clearUploadTimers();
    if (uploadAbortController) {
      try { uploadAbortController.abort(); } catch (e) { /* ignore */ }
      uploadAbortController = null;
    }
    if (uploadWs) {
      try { uploadWs.close(); } catch (e) { /* ignore */ }
      uploadWs = null;
    }
    uploadHttpResult = null;
    uploadWsDelivered = false;
    uploadFileId = null;
  }
  allSegments = [];
  demoData = null;
  audioManager.reset();
  resetResultsPanel();

  const btnStart = $('btnStart');
  if (btnStart) { btnStart.textContent = '演示'; btnStart.disabled = false; }

  if (transcriptBody) {
    transcriptBody.innerHTML = '';
  }
  if (emptyState) {
    emptyState.style.display = 'flex';
    const msgEl = emptyState.querySelector('div:nth-child(2)');
    if (msgEl) msgEl.textContent = '等待会议输入...';
  }

  if (logicPanel) logicPanel.innerHTML = '<div style="font-size:11px;color:var(--text-muted);text-align:center;padding:10px 0;">暂无数据</div>';
  if (processingPanel) processingPanel.innerHTML = '<div style="font-size:11px;color:var(--text-muted);text-align:center;padding:10px 0;">等待输入...</div>';
  if (summaryPanel) summaryPanel.innerHTML = '<div style="font-size:12px;color:var(--text-muted);text-align:center;">会议处理完成后将生成摘要...</div>';

  if (progressFill) progressFill.style.width = '0%';
  if (timeDisplay) timeDisplay.textContent = '--:--';
  if (segmentCounter) segmentCounter.textContent = '0/0';
  if (statSegments) statSegments.textContent = '0';
  if (statLogicFlags) statLogicFlags.textContent = '0';
  if (statLowConf) statLowConf.textContent = '0';
  if (statCorrections) statCorrections.textContent = '0';
  if (overallConfBar) overallConfBar.style.width = '0%';
  if (overallConfText) overallConfText.textContent = '--';
  if (meetingTitleDisplay) meetingTitleDisplay.textContent = '未加载会议';
  if (connectionStatus) connectionStatus.textContent = '离线';
  if (isRecording) stopMicRecording();
  else {
    stopMicVisualizer();
    setMicOrbVisible(false);
  }
}

// ═══════════════════════════════════════════════════════════
// Backend Status Check
// ═══════════════════════════════════════════════════════════

async function checkBackendStatus() {
  if (!xasrStatus) return;
  try {
    const resp = await fetch(`${API_BASE}/api/xasr/status`);
    const data = await resp.json();
    if (data.model_available) {
      xasrStatus.innerHTML = 'X-ASR 就绪';
      xasrStatus.style.color = 'var(--accent-green)';
    } else if (data.loading) {
      xasrStatus.innerHTML = 'X-ASR 加载中...';
      xasrStatus.style.color = 'var(--accent-yellow)';
    } else {
      xasrStatus.innerHTML = 'X-ASR 演示模式';
      xasrStatus.style.color = 'var(--accent-orange)';
    }
    updateConnectionStatus('connected');
  } catch (e) {
    xasrStatus.innerHTML = '后端离线';
    xasrStatus.style.color = 'var(--accent-red)';
    updateConnectionStatus('disconnected');
  }
}

// ═══════════════════════════════════════════════════════════
// Event Listeners (replaces inline onclick handlers)
// ═══════════════════════════════════════════════════════════
//
// NOTE: Because <script type="module"> is deferred, DOMContentLoaded
// has ALREADY fired by the time this code runs. We must NOT use
// addEventListener('DOMContentLoaded', ...) — it will never fire.
// Instead, we run initialization directly since the DOM is fully parsed.
// ═══════════════════════════════════════════════════════════

function initApp() {
  checkBackendStatus();
  connectLogStream();
  restoreUploadTaskOnLoad();

  // Button wiring
  const btnStart = $('btnStart');
  const btnReset = $('btnReset');
  const btnMic = $('btnMic');
  const btnUpload = $('btnUpload');
  const fileUpload = $('fileUpload');
  const btnLog = $('btnLog');
  const logPanelHeader = $('logPanelHeader');

  if (btnStart) btnStart.addEventListener('click', startDemo);
  if (btnReset) btnReset.addEventListener('click', resetAll);
  if (btnMic) btnMic.addEventListener('click', toggleMic);
  if (btnUpload) btnUpload.addEventListener('click', () => { if (fileUpload) fileUpload.click(); });
  if (fileUpload) fileUpload.addEventListener('change', handleFileUpload);
  if (btnLog) btnLog.addEventListener('click', toggleLogPanel);
  if (logPanelHeader) logPanelHeader.addEventListener('click', toggleLogPanel);

  console.log('[DiTing] App initialized — all event listeners registered');
}

// Run immediately — DOM is already parsed (module scripts are deferred)
initApp();

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.target.closest('input,textarea')) {
    e.preventDefault();
    if (!isPlaying) startDemo();
  }
  if (e.code === 'KeyR' && e.ctrlKey) {
    e.preventDefault();
    resetAll();
  }
  if (e.code === 'KeyL' && e.ctrlKey) {
    e.preventDefault();
    toggleLogPanel();
  }
});
