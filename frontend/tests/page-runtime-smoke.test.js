const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

function createElement(tagName = 'div') {
  const listeners = new Map();
  const classes = new Set();
  const element = {
    tagName: String(tagName).toUpperCase(),
    style: {},
    dataset: {},
    className: '',
    value: '',
    checked: false,
    disabled: false,
    hidden: false,
    textContent: '',
    innerHTML: '',
    children: [],
    options: [],
    classList: {
      add(...names) { names.forEach(name => classes.add(name)); },
      remove(...names) { names.forEach(name => classes.delete(name)); },
      toggle(name, force) {
        const enabled = force === undefined ? !classes.has(name) : Boolean(force);
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      },
      contains(name) { return classes.has(name); },
    },
    append(...items) { this.children.push(...items); },
    appendChild(item) { this.children.push(item); return item; },
    remove() {},
    setAttribute(name, value) { this[name] = String(value); },
    getAttribute(name) { return this[name] || ''; },
    addEventListener(type, handler) { listeners.set(type, handler); },
    removeEventListener(type) { listeners.delete(type); },
    dispatchEvent(event) {
      const handler = listeners.get(event.type);
      return handler ? handler(event) : true;
    },
    querySelector() { return createElement(); },
    querySelectorAll() { return []; },
    closest() { return null; },
    contains() { return false; },
    focus() {},
    click() { this.dispatchEvent({ type: 'click', target: this }); },
    getContext() {
      return {
        clearRect() {}, beginPath() {}, arc() {}, stroke() {}, fill() {},
        moveTo() {}, lineTo() {}, closePath() {}, fillRect() {},
        drawImage() {}, measureText() { return { width: 0 }; },
      };
    },
    toDataURL() { return 'data:image/jpeg;base64,'; },
  };
  return element;
}

const elements = new Map();
const document = {
  title: '',
  body: createElement('body'),
  documentElement: createElement('html'),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, createElement());
    return elements.get(id);
  },
  createElement,
  querySelector() { return createElement(); },
  querySelectorAll() { return []; },
  addEventListener() {},
  removeEventListener() {},
};

const context = {
  console,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  requestAnimationFrame: () => 0,
  cancelAnimationFrame() {},
  URLSearchParams,
  Blob,
  FileReader: class { readAsDataURL() { this.onload?.({ target: { result: '' } }); } },
  AudioContext: class { close() {} createMediaStreamSource() { return { connect() {} }; } createAnalyser() { return { fftSize: 0, smoothingTimeConstant: 0, frequencyBinCount: 1, getByteFrequencyData() {} }; } },
  WebSocket: class { constructor() { setTimeout(() => this.onopen?.({}), 0); } addEventListener() {} send() {} close() {} },
  navigator: {
    language: 'zh-CN',
    mediaDevices: {
      getUserMedia: () => Promise.reject(new Error('no media in smoke test')),
      enumerateDevices: () => Promise.resolve([]),
    },
  },
  location: { protocol: 'http:', hostname: 'localhost', search: '' },
  history: { pushState() {} },
  localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  addEventListener() {},
  removeEventListener() {},
  document,
  window: null,
  globalThis: null,
  fetch: () => Promise.resolve({
    ok: true,
    json: () => Promise.resolve({
      status: 'ok',
      service: '会悟 v2.0',
      api_revision: 'test',
      recognition: {},
      microphone: {},
      hotwords: { words: [] },
      models: { available_profiles: [] },
      meetings: [],
      users: [],
      reservations: [],
      analyses: [],
    }),
  }),
};
context.window = context;
context.globalThis = context;
context.window.webkitAudioContext = context.AudioContext;

vm.createContext(context);
for (const script of ['brand-storage.js', 'mic-level.js', 'live-protocol.js', 'management-transcription.js', 'app-settings.js', 'hotword-settings.js', 'i18n.js', 'js/live-mic-client.js', 'js/live-mic-controller.js', 'js/mic-orb.js', 'js/api-client.js', 'js/upload-controller.js', 'js/settings-controller.js', 'js/records-controller.js', 'js/app-init.js']) {
  vm.runInContext(fs.readFileSync(path.join(root, script), 'utf8'), context, { filename: script });
}

const moduleScript = html.match(/<script type="module">([\s\S]*?)<\/script>/);
assert.ok(moduleScript, 'inline module script should exist');

(async () => {
  const syntaxOnlyModule = `if (false) {\n${moduleScript[1]}\n}`;
  const syntaxOnlyUrl = `data:text/javascript;base64,${Buffer.from(syntaxOnlyModule).toString('base64')}`;
  await import(syntaxOnlyUrl);

  const executableScript = moduleScript[1].replace(
    /import\('https:\/\/unpkg\.com\/wavesurfer\.js@7\.8\/dist\/wavesurfer\.esm\.js'\)/g,
    'Promise.reject(new Error("dynamic import disabled"))',
  );
  assert.doesNotThrow(() => vm.runInContext(executableScript, context, { filename: 'index.html:inline-module' }));

  const settingsButton = document.getElementById('btnHotwords');
  const settingsModal = document.getElementById('hotwordModal');
  settingsModal.setAttribute('aria-hidden', 'true');
  await settingsButton.click();
  assert.equal(settingsModal.getAttribute('aria-hidden'), 'false');
  assert.equal(settingsModal.classList.contains('visible'), true);

  console.log('page runtime smoke test passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
