(function attachRecordsController(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuRecordsController = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createRecordsController() {
  function recordTitle(record) {
    return String(record?.title || record?.filename || record?.id || 'Untitled meeting');
  }
  return { recordTitle };
}));
