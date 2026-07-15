(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const HIDDEN_LOGIC_FLAG_TYPES = new Set(['self_contradiction']);

  function getVisibleLogicFlags(segment) {
    const flags = Array.isArray(segment?.logic_flags) ? segment.logic_flags : [];
    return flags.filter(flag => !HIDDEN_LOGIC_FLAG_TYPES.has(flag?.type));
  }

  return { getVisibleLogicFlags };
});
