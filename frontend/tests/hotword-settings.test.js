const assert = require('node:assert/strict');

const {
  buildPayload,
  getDomainPresets,
  getSelectedDomainPresetIds,
  normalizeSettings,
  scoreForNewWord,
  setDomainPresetSelection,
} = require('../hotword-settings.js');


const settings = normalizeSettings({
  enabled: true,
  fuzzy_pinyin_enabled: false,
  default_score: 6,
  words: [{ text: ' BERT ', score: 3, enabled: true }],
});

assert.equal(settings.words[0].text, 'BERT');
assert.equal(settings.words[0].score, 3);
assert.equal(settings.fuzzy_pinyin_enabled, false);
assert.equal(scoreForNewWord('OpenAI', 8), 2.5);
assert.equal(scoreForNewWord('贾扬清', 8), 8);

const payload = buildPayload(settings, [
  { text: 'OpenAI', score: 4.5, enabled: false },
]);
assert.equal(payload.words[0].score, 4.5);
assert.equal(payload.words[0].enabled, false);

const domainPresets = getDomainPresets();
assert.deepEqual(
  domainPresets.map(preset => preset.id),
  ['technology', 'product', 'business', 'finance', 'medical'],
);
assert.ok(domainPresets.every(preset => preset.name && preset.description && preset.word_count === 28));

const selectedPreset = setDomainPresetSelection([
  { text: 'bert', score: 9, enabled: false },
  { text: '自定义词', score: 4, enabled: true },
], 'technology', true, [], 8);
assert.equal(selectedPreset.selected, true);
assert.equal(selectedPreset.added_count, 27);
assert.equal(selectedPreset.changed_count, 28);
assert.deepEqual(selectedPreset.words[0], { text: 'bert', score: 9, enabled: true });
assert.equal(selectedPreset.words.find(word => word.text === '人工智能').score, 8);
assert.equal(selectedPreset.words.find(word => word.text === 'Transformer').score, 2.5);
assert.deepEqual(getSelectedDomainPresetIds(selectedPreset.words), ['technology']);

const selectedBusiness = setDomainPresetSelection([], 'business', true, [], 8);
const selectedFinance = setDomainPresetSelection(
  selectedBusiness.words,
  'finance',
  true,
  ['business'],
  8,
);
const deselectedFinance = setDomainPresetSelection(
  selectedFinance.words,
  'finance',
  false,
  ['business', 'finance'],
  8,
);
assert.equal(deselectedFinance.words.find(word => word.text === '净利润').enabled, false);
assert.equal(deselectedFinance.words.find(word => word.text === '预算').enabled, true);
assert.equal(deselectedFinance.words.find(word => word.text === '合规').enabled, true);
assert.deepEqual(getSelectedDomainPresetIds(deselectedFinance.words), ['business']);

console.log('hotword-settings tests passed');
