/** G2: LLM Settings — dialog, test connection, save.
 *
 *  Exposed on `window.DiTingLLMSettings`.
 */
(function attachLLMSettings(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingLLMSettings = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createLLMSettings() {

  const DEFAULT_PROVIDER = 'deepseek';
  const DEFAULT_MODEL = 'deepseek-v4-flash';
  const PROVIDER_PRESETS = {
    deepseek: { base_url: 'https://api.deepseek.com', default_model: DEFAULT_MODEL },
    qwen: { base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', default_model: 'qwen3.7-max' },
    openai: { base_url: 'https://api.openai.com/v1', default_model: 'gpt-5.2' },
    ollama: { base_url: 'http://127.0.0.1:11434/v1', default_model: '' },
    openai_compatible: { base_url: '', default_model: '' },
  };
  const MODEL_PRESETS = [
    { provider: 'deepseek', id: 'deepseek-v4-flash', label: 'DSv4 Flash（默认）', image_generation: false, diagram_mode: 'text' },
    { provider: 'deepseek', id: 'deepseek-v4-pro', label: 'DSv4 Pro', image_generation: false, diagram_mode: 'text' },
    { provider: 'qwen', id: 'qwen3.7-max', label: 'Qwen3.7 Max', image_generation: false, diagram_mode: 'text' },
    { provider: 'qwen', id: 'qwen3.7-plus', label: 'Qwen3.7 Plus', image_generation: false, diagram_mode: 'text' },
    { provider: 'qwen', id: 'qwen3.6-flash', label: 'Qwen3.6 Flash', image_generation: false, diagram_mode: 'text' },
    { provider: 'openai', id: 'gpt-5.2', label: 'GPT-5.2', image_generation: false, diagram_mode: 'text' },
    { provider: 'openai', id: 'gpt-5.1', label: 'GPT-5.1', image_generation: false, diagram_mode: 'text' },
    { provider: 'openai', id: 'gpt-5-mini', label: 'GPT-5 mini', image_generation: false, diagram_mode: 'text' },
  ];
  let activeCatalog = MODEL_PRESETS.slice();

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

  function resolveApiBase() {
    const params = new URLSearchParams(window.location.search);
    const host = params.get('apiHost') || window.location.hostname;
    const port = params.get('apiPort') || '8765';
    const pageProto = window.location.protocol.replace(':', '');
    const proto = params.get('apiProto') || (['http', 'https'].includes(pageProto) ? pageProto : 'http');
    return `${proto}://${host}:${port}`;
  }

  function providerPreset(provider) {
    return PROVIDER_PRESETS[provider] || PROVIDER_PRESETS.openai_compatible;
  }

  function mergeModelOptions(primary, secondary) {
    return [...new Set([...(primary || []), ...(secondary || [])].filter(Boolean))];
  }

  function renderModelOptions(models) {
    const datalist = $('llmModelOptions');
    if (!datalist || typeof document === 'undefined') return;
    datalist.textContent = '';
    for (const model of models) {
      const item = typeof model === 'string' ? { id: model, label: model } : model;
      if (!item?.id) continue;
      const option = document.createElement('option');
      option.value = item.id;
      option.label = item.label || item.id;
      datalist.appendChild(option);
    }
  }

  function updateCapabilityNote() {
    const note = $('llmCapabilityNote');
    if (!note) return;
    const modelName = $('llmModelName')?.value || DEFAULT_MODEL;
    note.textContent = `${modelName} 按文字模型调用；流程图输出为 Mermaid + Markdown，不调用图像生成接口。`;
  }

  function applyProviderPreset(force = true) {
    const provider = $('llmProvider')?.value || DEFAULT_PROVIDER;
    const preset = providerPreset(provider);
    const baseUrl = $('llmBaseUrl');
    const modelName = $('llmModelName');
    if (force || !baseUrl?.value) {
      if (baseUrl) baseUrl.value = preset.base_url;
    }
    if (force || !modelName?.value) {
      if (modelName) modelName.value = preset.default_model;
    }
    renderModelOptions(activeCatalog.filter((item) => !item.provider || item.provider === provider));
    updateCapabilityNote();
  }

  function requestFormData() {
    const data = collectFormData();
    if (!data.api_key) delete data.api_key;
    return data;
  }

  // ── Open / Close ────────────────────────────────────────────

  async function open() {
    const dialog = $('llmSettingsDialog');
    if (!dialog) return;
    dialog.style.display = 'flex';
    dialog.setAttribute('aria-hidden', 'false');

    // Load existing settings
    try {
      const apiBase = resolveApiBase();
      const data = await apiFetch(`${apiBase}/api/llm-settings`);
      const s = data.settings || {};
      if (Array.isArray(data.catalog?.models)) activeCatalog = data.catalog.models;
      const provider = $('llmProvider');
      const baseUrl = $('llmBaseUrl');
      const modelName = $('llmModelName');
      const apiKeyHint = $('llmApiKeyHint');
      if (provider) provider.value = s.provider || DEFAULT_PROVIDER;
      if (baseUrl) baseUrl.value = s.base_url || providerPreset(provider?.value).base_url;
      if (modelName) modelName.value = s.model_name || providerPreset(provider?.value).default_model || DEFAULT_MODEL;
      renderModelOptions(activeCatalog.filter((item) => !item.provider || item.provider === provider?.value));
      if (apiKeyHint) apiKeyHint.textContent = s.has_api_key ? '(已保存，留空则保留原密钥)' : '(未保存密钥)';
      const apiKey = $('llmApiKey');
      if (apiKey) apiKey.value = '';
      const temp = $('llmTemperature');
      if (temp) temp.value = s.temperature ?? 0.2;
      const maxTok = $('llmMaxTokens');
      if (maxTok) maxTok.value = s.max_tokens ?? 8192;
      const timeout = $('llmTimeout');
      if (timeout) timeout.value = s.timeout_sec ?? 180;
      const diag = $('llmDiagramEnabled');
      if (diag) diag.checked = s.diagram_enabled !== false;
      const diagType = $('llmDiagramType');
      if (diagType) diagType.value = s.diagram_type || 'auto';
      const lang = $('llmOutputLang');
      if (lang) lang.value = s.output_language || 'zh-CN';
      const formal = $('llmFormalStyle');
      if (formal) formal.checked = s.formal_style !== false;
      const formula = $('llmFormulaMode');
      if (formula) formula.value = s.formula_mode || 'latex';
      updateCapabilityNote();
    } catch (e) {
      const provider = $('llmProvider');
      if (provider) provider.value = DEFAULT_PROVIDER;
      applyProviderPreset(true);
    }
  }

  function close() {
    const dialog = $('llmSettingsDialog');
    if (dialog) { dialog.style.display = 'none'; dialog.setAttribute('aria-hidden', 'true'); }
  }

  // ── Collect form data ───────────────────────────────────────

  function collectFormData() {
    return {
      provider: $('llmProvider')?.value || '',
      base_url: $('llmBaseUrl')?.value || '',
      model_name: $('llmModelName')?.value || '',
      api_key: $('llmApiKey')?.value || '',
      temperature: parseFloat($('llmTemperature')?.value || '0.2'),
      max_tokens: parseInt($('llmMaxTokens')?.value || '8192', 10),
      timeout_sec: parseInt($('llmTimeout')?.value || '180', 10),
      diagram_enabled: $('llmDiagramEnabled')?.checked !== false,
      diagram_type: $('llmDiagramType')?.value || 'auto',
      output_language: $('llmOutputLang')?.value || 'zh-CN',
      formal_style: $('llmFormalStyle')?.checked !== false,
      formula_mode: $('llmFormulaMode')?.value || 'latex',
    };
  }

  // ── Actions ─────────────────────────────────────────────────

  async function testConnection() {
    const statusEl = $('llmTestStatus');
    const btn = $('btnTestLLM');
    if (statusEl) statusEl.textContent = '测试中...';
    if (btn) btn.disabled = true;
    try {
      const apiBase = resolveApiBase();
      const formData = requestFormData();
      const data = await apiFetch(`${apiBase}/api/llm-settings/test`, {
        method: 'POST',
        body: JSON.stringify(formData),
      });
      if (data.ok) {
        if (statusEl) { statusEl.textContent = '✓ 连接成功'; statusEl.style.color = 'var(--accent-green)'; }
      } else {
        if (statusEl) { statusEl.textContent = '✗ ' + (data.message || '连接失败'); statusEl.style.color = 'var(--accent-red)'; }
      }
    } catch (e) {
      if (statusEl) { statusEl.textContent = '✗ ' + (e.message || '测试失败'); statusEl.style.color = 'var(--accent-red)'; }
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function refreshModels() {
    const statusEl = $('llmTestStatus');
    const button = $('btnRefreshLLMModels');
    if (statusEl) statusEl.textContent = '正在读取模型列表...';
    if (button) button.disabled = true;
    try {
      const apiBase = resolveApiBase();
      const data = await apiFetch(`${apiBase}/api/llm-settings/models`, {
        method: 'POST',
        body: JSON.stringify(requestFormData()),
      });
      if (!data.ok) throw new Error(data.message || '读取失败');
      const provider = $('llmProvider')?.value || DEFAULT_PROVIDER;
      const curated = activeCatalog.filter((item) => !item.provider || item.provider === provider);
      const merged = mergeModelOptions(data.models, curated.map((item) => item.id));
      renderModelOptions(merged);
      if (statusEl) { statusEl.textContent = `✓ 已读取 ${merged.length} 个模型`; statusEl.style.color = 'var(--accent-green)'; }
    } catch (e) {
      if (statusEl) { statusEl.textContent = '✗ ' + (e.message || '读取失败'); statusEl.style.color = 'var(--accent-red)'; }
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function saveSettings() {
    const statusEl = $('llmSaveStatus');
    try {
      const apiBase = resolveApiBase();
      const formData = requestFormData();
      await apiFetch(`${apiBase}/api/llm-settings`, {
        method: 'PUT',
        body: JSON.stringify(formData),
      });
      if (statusEl) { statusEl.textContent = '✓ 已保存'; statusEl.style.color = 'var(--accent-green)'; }
    } catch (e) {
      if (statusEl) { statusEl.textContent = '✗ 保存失败: ' + (e.message || ''); statusEl.style.color = 'var(--accent-red)'; }
    }
  }

  return {
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    MODEL_PRESETS,
    applyProviderPreset,
    close,
    collectFormData,
    mergeModelOptions,
    open,
    providerPreset,
    refreshModels,
    saveSettings,
    testConnection,
    updateCapabilityNote,
  };
}));
