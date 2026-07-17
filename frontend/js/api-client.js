(function attachApiClient(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuApiClient = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createApiClient(root) {
  function createApiClient({ apiBase = '', wsBase = '', fetchImpl } = {}) {
    const fetcher = fetchImpl || root?.fetch;
    return {
      apiBase,
      wsBase,
      async getJson(path) {
        const response = await fetcher(`${apiBase}${path}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      },
      async postJson(path, body) {
        const response = await fetcher(`${apiBase}${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body || {}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data?.detail || data?.error || `HTTP ${response.status}`);
        return data;
      },
      uploadStatus(fileId) { return this.getJson(`/api/audio/upload/${fileId}/status`); },
      cancelUpload(fileId) { return this.postJson(`/api/audio/upload/${fileId}/cancel`, {}); },
    };
  }
  return { createApiClient };
}));
