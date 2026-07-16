/** S6: Record Summary — dialog, polling, preview, download.
 *
 *  Exposed on `window.DiTingRecordSummary` so the records panel can
 *  call `DiTingRecordSummary.open(recordIds)`.
 */
(function attachRecordSummary(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingRecordSummary = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createRecordSummary() {

  let currentSummaryId = null;
  let pollingTimer = null;
  let currentRecordIds = [];

  // ── Helpers ──────────────────────────────────────────────────

  function $(id) { return document.getElementById(id); }

  function show(el) { if (el) el.style.display = ''; }
  function hide(el) { if (el) el.style.display = 'none'; }

  async function apiFetch(url, opts) {
    const token = localStorage.getItem('diting_auth_token') || '';
    const headers = { 'Content-Type': 'application/json', ...(opts?.headers || {}) };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const resp = await fetch(url, { ...opts, headers });
    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(body || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  // Resolve API base (same logic as management-transcription.js)
  function resolveApiBase() {
    const params = new URLSearchParams(window.location.search);
    const host = params.get('apiHost') || window.location.hostname;
    const port = params.get('apiPort') || '8765';
    const pageProto = window.location.protocol.replace(':', '');
    const proto = params.get('apiProto') || (pageProto === 'http' || pageProto === 'https' ? pageProto : 'http');
    return `${proto}://${host}:${port}`;
  }

  // ── Open dialog ──────────────────────────────────────────────

  function open(recordIds) {
    if (!recordIds || !recordIds.length) return;
    currentRecordIds = recordIds.slice();
    currentSummaryId = null;
    stopPolling();

    const dialog = $('summaryDialog');
    const configForm = $('summaryConfigForm');
    const resultArea = $('summaryResultArea');
    const titleInput = $('summaryTitleInput');
    const typeSelect = $('summaryTypeSelect');
    const recordList = $('summaryRecordList');

    if (dialog) { dialog.style.display = 'flex'; dialog.setAttribute('aria-hidden', 'false'); }
    if (configForm) show(configForm);
    if (resultArea) hide(resultArea);
    if (titleInput) titleInput.value = recordIds.length > 1 ? '多会议综合摘要' : '会议摘要';
    if (typeSelect) typeSelect.value = recordIds.length > 1 ? 'comprehensive' : 'standard';
    if (recordList) recordList.textContent = `${recordIds.length} 条记录 (${recordIds.join(', ').slice(0, 120)}...)`;

    // Update change-analysis checkbox
    const cbChanges = $('summaryOptChanges');
    if (cbChanges) {
      cbChanges.checked = recordIds.length > 1;
      cbChanges.disabled = recordIds.length <= 1;
    }

    // Set default button label
    const btnSubmit = $('btnSubmitSummary');
    if (btnSubmit) {
      btnSubmit.textContent = recordIds.length > 1 ? '开始生成综合摘要' : '开始生成摘要';
    }
  }

  function close() {
    stopPolling();
    const dialog = $('summaryDialog');
    if (dialog) { dialog.style.display = 'none'; dialog.setAttribute('aria-hidden', 'true'); }
  }

  // ── Submit ───────────────────────────────────────────────────

  async function submit() {
    const titleInput = $('summaryTitleInput');
    const typeSelect = $('summaryTypeSelect');
    const langSelect = $('summaryLangSelect');
    const title = (titleInput?.value || '未命名摘要').trim();
    const summaryType = typeSelect?.value || 'standard';
    const language = langSelect?.value || 'zh-CN';

    const options = {
      include_speakers: $('summaryOptSpeakers')?.checked ?? true,
      include_decisions: $('summaryOptDecisions')?.checked ?? true,
      include_action_items: $('summaryOptActions')?.checked ?? true,
      compare_changes: $('summaryOptChanges')?.checked ?? false,
    };

    const configForm = $('summaryConfigForm');
    const resultArea = $('summaryResultArea');
    const progressEl = $('summaryProgress');

    try {
      hide(configForm);
      show(resultArea);
      if (progressEl) progressEl.textContent = '正在创建摘要任务...';

      const apiBase = resolveApiBase();
      const resp = await apiFetch(`${apiBase}/api/record-summaries`, {
        method: 'POST',
        body: JSON.stringify({
          record_ids: currentRecordIds,
          title,
          summary_type: summaryType,
          language,
          options,
        }),
      });

      currentSummaryId = resp.summary.id;
      if (progressEl) progressEl.textContent = '摘要任务已创建，正在等待 LLM 生成...';
      startPolling();
    } catch (err) {
      if (progressEl) progressEl.textContent = '创建摘要失败：' + (err.message || '未知错误');
      show($('btnRetrySummary'));
    }
  }

  // ── Polling ──────────────────────────────────────────────────

  function startPolling() {
    stopPolling();
    pollStatus();
    pollingTimer = setInterval(pollStatus, 2000);
  }

  function stopPolling() {
    if (pollingTimer) { clearInterval(pollingTimer); pollingTimer = null; }
  }

  async function pollStatus() {
    if (!currentSummaryId) return;
    try {
      const apiBase = resolveApiBase();
      const data = await apiFetch(`${apiBase}/api/record-summaries/${encodeURIComponent(currentSummaryId)}`);
      const summary = data.summary;
      const progressEl = $('summaryProgress');
      const previewEl = $('summaryMarkdownPreview');
      const btnCopy = $('btnCopySummary');
      const btnDownload = $('btnDownloadSummary');
      const btnRetry = $('btnRetrySummary');

      if (summary.status === 'processing') {
        if (progressEl) progressEl.textContent =
          `生成中... ${summary.stage || ''} (${Math.round((summary.progress || 0) * 100)}%)`;
      } else if (summary.status === 'completed') {
        stopPolling();
        if (progressEl) progressEl.textContent = '生成完成！';
        if (previewEl) previewEl.textContent = summary.markdown_content || '（无内容）';
        show(btnCopy);
        show(btnDownload);
        hide(btnRetry);
      } else if (summary.status === 'failed') {
        stopPolling();
        if (progressEl) progressEl.textContent = '生成失败：' + (summary.error_message || '未知错误');
        show(btnRetry);
        hide(btnCopy);
        hide(btnDownload);
      }
    } catch (err) {
      // Polling error is non-fatal; next tick will retry
    }
  }

  // ── Actions ──────────────────────────────────────────────────

  async function retry() {
    if (!currentSummaryId) return;
    try {
      const apiBase = resolveApiBase();
      await apiFetch(`${apiBase}/api/record-summaries/${encodeURIComponent(currentSummaryId)}/retry`, {
        method: 'POST',
      });
      const configForm = $('summaryConfigForm');
      const resultArea = $('summaryResultArea');
      hide(configForm);
      show(resultArea);
      const progressEl = $('summaryProgress');
      if (progressEl) progressEl.textContent = '正在重新生成...';
      hide($('btnRetrySummary'));
      startPolling();
    } catch (err) {
      const progressEl = $('summaryProgress');
      if (progressEl) progressEl.textContent = '重试失败：' + (err.message || '未知错误');
    }
  }

  async function downloadMarkdown() {
    if (!currentSummaryId) return;
    try {
      const apiBase = resolveApiBase();
      const resp = await fetch(
        `${apiBase}/api/record-summaries/${encodeURIComponent(currentSummaryId)}/download`,
        { headers: authHeaders() },
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      const disposition = resp.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?(.+?)"?$/);
      anchor.download = match ? match[1] : 'meeting-summary.md';
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('下载失败：' + (err.message || '未知错误'));
    }
  }

  async function copyMarkdown() {
    const previewEl = $('summaryMarkdownPreview');
    if (!previewEl) return;
    try {
      await navigator.clipboard.writeText(previewEl.textContent || '');
      alert('已复制到剪贴板');
    } catch (_) {
      // Fallback
      const ta = document.createElement('textarea');
      ta.value = previewEl.textContent || '';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
  }

  function backToConfig() {
    stopPolling();
    show($('summaryConfigForm'));
    hide($('summaryResultArea'));
  }

  function authHeaders() {
    const token = localStorage.getItem('diting_auth_token') || '';
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  // ── Public API ───────────────────────────────────────────────

  return {
    open,
    close,
    submit,
    retry,
    downloadMarkdown,
    copyMarkdown,
    backToConfig,
    getSummaryId: () => currentSummaryId,
    getRecordIds: () => currentRecordIds,
  };
}));
