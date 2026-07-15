(function attachHotwordSettings(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingHotwordSettings = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createHotwordSettings() {
  const clamp = (value, fallback = 5) => {
    const parsed = Number(value);
    const safe = Number.isFinite(parsed) ? parsed : fallback;
    return Math.round(Math.min(20, Math.max(0.1, safe)) * 1000) / 1000;
  };

  const containsCjk = value => /[\u3400-\u9fff\uf900-\ufaff]/.test(value || '');

  function scoreForNewWord(text, defaultScore) {
    const score = clamp(defaultScore);
    return containsCjk(text) ? score : Math.min(score, 2.5);
  }

  function normalizeSettings(payload = {}) {
    const defaultScore = clamp(payload.default_score);
    const words = Array.isArray(payload.words) ? payload.words : [];
    return {
      enabled: payload.enabled !== false,
      fuzzy_pinyin_enabled: payload.fuzzy_pinyin_enabled !== false,
      default_score: defaultScore,
      words: words.map(item => ({
        text: String(item?.text || '').trim(),
        score: clamp(item?.score, scoreForNewWord(item?.text, defaultScore)),
        enabled: item?.enabled !== false,
      })).filter(item => item.text),
    };
  }

  function buildPayload(settings, rows) {
    return normalizeSettings({
      ...settings,
      words: rows,
    });
  }

  return { clamp, containsCjk, scoreForNewWord, normalizeSettings, buildPayload };
}));
