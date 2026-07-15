const assert = require('node:assert/strict');

const {
  buildAnalysisPayload,
  buildUploadUrl,
  resolveBackend,
} = require('../management-transcription.js');


const backend = resolveBackend({
  protocol: 'http:',
  hostname: '127.0.0.1',
  search: '?apiPort=8766',
});
assert.deepEqual(backend, {
  apiBase: 'http://127.0.0.1:8766',
  wsBase: 'ws://127.0.0.1:8766',
});

const uploadUrl = new URL(buildUploadUrl(backend.apiBase, 'file id', {
  enableDiarization: true,
  numSpeakers: 3,
}));
assert.equal(uploadUrl.pathname, '/api/audio/upload');
assert.equal(uploadUrl.searchParams.get('file_id'), 'file id');
assert.equal(uploadUrl.searchParams.get('enable_diarization'), 'true');
assert.equal(uploadUrl.searchParams.get('num_speakers'), '3');

const analysis = buildAnalysisPayload({
  meetingId: 'meeting-1',
  title: 'Weekly sync.wav',
  segments: [
    {
      display_text: '项目按计划推进',
      start_sec: 0,
      end_sec: 2.5,
      speaker_id: 'SPEAKER_00',
      speaker_name: '张三',
      corrections: [{ original: '相木', corrected: '项目' }],
    },
  ],
});
assert.equal(analysis.meeting_id, 'meeting-1');
assert.equal(analysis.transcript_json[0].speaker, '张三');
assert.equal(analysis.duration_sec, 2.5);
assert.equal(analysis.corrections_count, 1);
assert.equal(analysis.overall_confidence, 0);

console.log('management transcription tests passed');
