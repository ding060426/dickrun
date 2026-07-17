(function attachUploadController(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuUploadController = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createUploadController() {
  function normalizeUploadJob(payload) {
    const source = payload && typeof payload === 'object' ? payload : {};
    return {
      file_id: String(source.file_id || ''),
      filename: String(source.filename || ''),
      status: String(source.status || 'unknown'),
      progress: Math.max(0, Math.min(1, Number(source.progress || 0))),
      error: source.error || null,
      result: source.result || null,
    };
  }
  return { normalizeUploadJob };
}));
