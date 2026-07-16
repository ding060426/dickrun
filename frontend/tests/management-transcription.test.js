const assert = require('node:assert/strict');

const {
  buildAnalysisPayload,
  buildRecordPayload,
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

const record = buildRecordPayload({
  recordId: 'record-1',
  meetingId: 'meeting-1',
  title: '周会记录',
  sourceType: 'upload',
  sourceFilename: 'weekly.wav',
  sourceMimeType: 'audio/wav',
  sourceSizeBytes: 2048,
  speakers: [{ id: 'SPEAKER_00', name: '张三' }],
  segments: [{
    index: 1,
    text: '项目按计划推进',
    raw_text: '项目按计划推进',
    start_sec: 0,
    end_sec: 2.5,
    speaker_id: 'SPEAKER_00',
    speaker_name: '张三',
    audio_wav_base64: 'UklGRg==',
    asr_confidence: 0.93,
  }],
});
assert.equal(record.id, 'record-1');
assert.equal(record.source_type, 'upload');
assert.equal(record.segments[0].audio_wav_base64, 'UklGRg==');
assert.match(record.full_text, /张三.*项目按计划推进/);
assert.deepEqual(record.speakers, [{ id: 'SPEAKER_00', name: '张三' }]);

console.log('management transcription tests passed');
