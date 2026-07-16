(function attachBrandStorage(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuStorage = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createBrandStorage() {
  const AUTH_TOKEN_KEY = 'huiwu:auth-token';
  const LEGACY_AUTH_TOKEN_KEY = 'diting_auth_token';

  function resolveStorage(storage) {
    if (storage) return storage;
    try { return globalThis.localStorage; } catch (error) { return null; }
  }

  function getAuthToken(storage) {
    const target = resolveStorage(storage);
    if (!target) return '';
    try {
      const current = target.getItem(AUTH_TOKEN_KEY) || '';
      if (current) return current;
      const legacy = target.getItem(LEGACY_AUTH_TOKEN_KEY) || '';
      if (!legacy) return '';
      target.setItem(AUTH_TOKEN_KEY, legacy);
      target.removeItem(LEGACY_AUTH_TOKEN_KEY);
      return legacy;
    } catch (error) {
      return '';
    }
  }

  function setAuthToken(token, storage) {
    const target = resolveStorage(storage);
    if (!target) return;
    try {
      if (token) target.setItem(AUTH_TOKEN_KEY, String(token));
      else target.removeItem(AUTH_TOKEN_KEY);
      target.removeItem(LEGACY_AUTH_TOKEN_KEY);
    } catch (error) { /* ignore unavailable storage */ }
  }

  function clearAuthToken(storage) {
    const target = resolveStorage(storage);
    if (!target) return;
    try {
      target.removeItem(AUTH_TOKEN_KEY);
      target.removeItem(LEGACY_AUTH_TOKEN_KEY);
    } catch (error) { /* ignore unavailable storage */ }
  }

  return { AUTH_TOKEN_KEY, getAuthToken, setAuthToken, clearAuthToken };
}));
