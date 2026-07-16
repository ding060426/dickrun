const assert = require('node:assert/strict');

const {
  DEFAULT_MODEL,
  DEFAULT_PROVIDER,
  MODEL_PRESETS,
  mergeModelOptions,
  providerPreset,
} = require('../llm-settings.js');

assert.equal(DEFAULT_PROVIDER, 'deepseek');
assert.equal(DEFAULT_MODEL, 'deepseek-v4-flash');
assert.equal(providerPreset('deepseek').base_url, 'https://api.deepseek.com');
assert.ok(MODEL_PRESETS.some((item) => item.id === 'deepseek-v4-pro'));
assert.ok(MODEL_PRESETS.every((item) => item.diagram_mode === 'text'));
assert.deepEqual(
  mergeModelOptions(['deepseek-v4-pro', 'custom-dsv4pro'], ['deepseek-v4-pro']),
  ['deepseek-v4-pro', 'custom-dsv4pro'],
);

console.log('llm-settings tests passed');
