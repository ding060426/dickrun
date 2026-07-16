(function attachPresentation(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuPresentation = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createPresentation() {
  function formatDuration(segment = {}) {
    const start = Number(segment.start_sec ?? segment.start);
    const end = Number(segment.end_sec ?? segment.end);
    if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
      return `${(end - start).toFixed(1)}s`;
    }
    const duration = Number(segment.duration_sec ?? segment.duration);
    return Number.isFinite(duration) && duration >= 0 ? `${duration.toFixed(1)}s` : '--';
  }

  return { formatDuration };
}));
