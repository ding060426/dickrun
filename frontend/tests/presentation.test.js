const assert = require('node:assert/strict');

const {
  confidenceTier,
  formatConfidence,
  formatDuration,
  summarizeConfidence,
} = require('../presentation.js');

assert.equal(formatDuration({ start_sec: 1.2, end_sec: 4.7 }), '3.5s');
assert.equal(formatDuration({ duration: 2 }), '2.0s');
assert.equal(formatDuration({}), '--');
assert.equal(formatConfidence(0.826), '83%');
assert.equal(formatConfidence(82.6), '83%');
assert.equal(formatConfidence(null), '--');
assert.equal(confidenceTier(0.8), 'high');
assert.equal(confidenceTier(0.6), 'medium');
assert.equal(confidenceTier(0.3), 'low');

assert.deepEqual(
  summarizeConfidence([
    { asr_confidence: 0.9 },
    { confidence: 0.7 },
    { text: 'missing confidence is ignored' },
  ]),
  { average: 0.8, percent: 80, tier: 'high' },
);
assert.equal(summarizeConfidence([{ text: 'no confidence' }]), null);

console.log('presentation tests passed');
