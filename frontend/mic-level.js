(function attachMicLevel(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingMicLevel = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createMicLevel() {
  function clamp(value, minimum = 0, maximum = 1) {
    return Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : 0));
  }

  function nextMicLevel(previous, rms, peak) {
    const current = clamp(previous);
    const target = clamp(Math.max(clamp(rms) * 8, clamp(peak) * 1.8));
    const smoothing = target > current ? 0.65 : 0.14;
    return clamp(current + (target - current) * smoothing);
  }

  function buildBarScales(level, phase, count = 11) {
    const normalized = clamp(level);
    const size = Math.max(1, Math.floor(count));
    const center = (size - 1) / 2;
    return Array.from({ length: size }, (_, index) => {
      const distance = center ? Math.abs(index - center) / center : 0;
      const envelope = 0.28 + (1 - distance) * 0.72;
      const motion = 0.58 + Math.abs(Math.sin(phase + index * 0.74)) * 0.42;
      return clamp(0.12 + normalized * envelope * motion, 0.12, 1);
    });
  }

  return { nextMicLevel, buildBarScales };
}));
