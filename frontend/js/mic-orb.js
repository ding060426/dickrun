(function attachMicOrb(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuMicOrb = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createMicOrb(root) {
  function clampPanelPosition(x, y, viewport, rect) {
    const w = rect?.width || rect?.w || 240;
    const h = rect?.height || rect?.h || 150;
    return {
      x: Math.max(0, Math.min(Number(x) || 0, Math.max(0, viewport.width - w))),
      y: Math.max(0, Math.min(Number(y) || 0, Math.max(0, viewport.height - h))),
    };
  }

  function detectDockSide(x, viewport, margin = 24) {
    if (x <= margin) return 'left';
    if (x >= viewport.width - margin) return 'right';
    return null;
  }

  class MicOrb {
    constructor({ panel, wrap, wave, title, subtitle, dockTab, levelApi } = {}) {
      this.panel = panel;
      this.wrap = wrap;
      this.wave = wave;
      this.title = title;
      this.subtitle = subtitle;
      this.dockTab = dockTab;
      this.levelApi = levelApi || root?.HuiWuMicLevel;
      this.level = 0;
      this.docked = false;
      this.dockSide = null;
    }

    setState(state) {
      const active = state === 'recording' || state === 'finalizing';
      this.panel?.classList?.toggle('visible', active && !this.docked);
      this.panel?.classList?.toggle('listening', state === 'recording');
      this.panel?.classList?.toggle('finalizing', state === 'finalizing');
      this.dockTab?.classList?.toggle('visible', active && this.docked);
      if (!active) this.setLevel(0, 0);
    }

    setMessage(title, subtitle) {
      if (this.title && title) this.title.textContent = title;
      if (this.subtitle && subtitle) this.subtitle.textContent = subtitle;
    }

    setLevel(rms, peak) {
      if (!this.levelApi) return;
      this.level = this.levelApi.nextMicLevel(this.level, rms, peak);
      const bars = Array.from(this.wave?.querySelectorAll?.('span') || []);
      const scales = this.levelApi.buildBarScales(this.level, 0, bars.length);
      bars.forEach((bar, index) => { bar.style.transform = `scaleY(${scales[index] || 0.12})`; });
      const orb = this.levelApi.buildOrbStyle(this.level);
      if (this.wrap && orb) {
        this.wrap.style.setProperty('--orb-scale', orb.scale.toFixed(3));
        this.wrap.style.setProperty('--orb-core-scale', orb.coreScale.toFixed(3));
      }
    }

    dock(side = 'right') {
      this.docked = true;
      this.dockSide = side;
      this.panel?.classList?.add('docked');
      this.dockTab?.classList?.add('visible');
    }

    undock() {
      this.docked = false;
      this.dockSide = null;
      this.panel?.classList?.remove('docked');
      this.dockTab?.classList?.remove('visible');
    }
  }

  return { MicOrb, clampPanelPosition, detectDockSide };
}));
