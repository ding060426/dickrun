(function attachPresentation(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingPresentation = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createPresentation() {
  function normalizeConfidence(value) {
    if (value === undefined || value === null || value === '') return null;
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    const normalized = number > 1 ? number / 100 : number;
    return Math.max(0, Math.min(1, normalized));
  }

  function formatDuration(segment = {}) {
    const start = Number(segment.start_sec ?? segment.start);
    const end = Number(segment.end_sec ?? segment.end);
    if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
      return `${(end - start).toFixed(1)}s`;
    }
    const duration = Number(segment.duration_sec ?? segment.duration);
    return Number.isFinite(duration) && duration >= 0 ? `${duration.toFixed(1)}s` : '--';
  }

  function formatConfidence(value) {
    const normalized = normalizeConfidence(value);
    return normalized === null ? '--' : `${Math.round(normalized * 100)}%`;
  }

  function confidenceTier(value) {
    const normalized = normalizeConfidence(value);
    if (normalized === null) return 'unknown';
    if (normalized >= 0.75) return 'high';
    if (normalized >= 0.5) return 'medium';
    return 'low';
  }

  function summarizeConfidence(segments) {
    const values = (Array.isArray(segments) ? segments : [])
      .map(segment => normalizeConfidence(segment?.asr_confidence ?? segment?.confidence))
      .filter(value => value !== null);
    if (values.length === 0) return null;
    const average = Math.round((values.reduce((sum, value) => sum + value, 0) / values.length) * 1000) / 1000;
    return {
      average,
      percent: Math.round(average * 100),
      tier: confidenceTier(average),
    };
  }

  return {
    confidenceTier,
    formatConfidence,
    formatDuration,
    normalizeConfidence,
    summarizeConfidence,
  };
}));
