const assert = require('node:assert/strict');

const {
  applyDomainPreset,
  buildPayload,
  getDomainPresets,
  normalizeSettings,
  scoreForNewWord,
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
assert.ok(domainPresets.every(preset => preset.name && preset.description && preset.word_count >= 10));

const appliedPreset = applyDomainPreset([
  { text: 'bert', score: 9, enabled: false },
  { text: '自定义词', score: 4, enabled: true },
], 'technology', 8);
assert.equal(appliedPreset.added_count, 11);
assert.equal(appliedPreset.existing_count, 1);
assert.deepEqual(appliedPreset.words[0], { text: 'bert', score: 9, enabled: false });
assert.equal(appliedPreset.words.find(word => word.text === '人工智能').score, 8);
assert.equal(appliedPreset.words.find(word => word.text === 'Transformer').score, 2.5);

console.log('hotword-settings tests passed');
