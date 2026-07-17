const assert = require('node:assert/strict');
const { modelStatusSummary } = require('../js/settings-controller.js');

const summary = modelStatusSummary({
  profiles: { live: { effective_profile: 'low-latency' }, final: { effective_profile: 'low-latency' } },
  diarization: { available: false },
  providers: { qwen3: { mode: 'final_transcription_only' } },
});
assert.equal(summary.live_profile, 'low-latency');
assert.equal(summary.diarization_mode, 'asr_only');
console.log('settings controller tests passed');
