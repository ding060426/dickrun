const assert = require('node:assert/strict');
const { normalizeUploadJob } = require('../js/upload-controller.js');

const job = normalizeUploadJob({ file_id: 123, filename: 'a.wav', status: 'recognizing', progress: 5 });
assert.equal(job.file_id, '123');
assert.equal(job.filename, 'a.wav');
assert.equal(job.progress, 1);
console.log('upload controller tests passed');
