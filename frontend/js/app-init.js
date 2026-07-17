(function attachAppInit(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuAppInit = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createAppInit() {
  function ready(callback) {
    if (typeof document === 'undefined') return callback();
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', callback, { once: true });
    else callback();
  }
  return { ready };
}));
