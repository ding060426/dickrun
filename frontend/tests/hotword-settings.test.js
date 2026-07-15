const assert = require('node:assert/strict');

const {
  buildPayload,
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

console.log('hotword-settings tests passed');
