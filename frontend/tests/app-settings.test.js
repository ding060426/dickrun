const assert = require('node:assert/strict');

const {
  buildMediaConstraints,
  normalizeAppSettings,
} = require('../app-settings.js');


const settings = normalizeAppSettings({
  recognition: {
    live_asr_profile: 'quality',
    final_asr_profile: 'invalid',
    file_vad_threshold: 4,
  },
  microphone: {
    device_id: 'mic-123',
    echo_cancellation: false,
    noise_suppression: true,
    auto_gain_control: true,
    endpoint_grace_ms: 9000,
  },
});

assert.equal(settings.recognition.live_asr_profile, 'quality');
assert.equal(settings.recognition.final_asr_profile, 'meeting');
assert.equal(settings.recognition.file_vad_threshold, 0.95);
assert.equal(settings.microphone.endpoint_grace_ms, 5000);

const constraints = buildMediaConstraints(settings.microphone);
assert.deepEqual(constraints, {
  audio: {
    channelCount: 1,
    deviceId: { exact: 'mic-123' },
    echoCancellation: false,
    noiseSuppression: true,
    autoGainControl: true,
  },
});

console.log('app-settings tests passed');
