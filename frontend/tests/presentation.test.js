const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const presentation = require('../presentation.js');
const { formatDuration } = presentation;

assert.deepEqual(Object.keys(presentation), ['formatDuration']);

assert.equal(formatDuration({ start_sec: 1.2, end_sec: 4.7 }), '3.5s');
assert.equal(formatDuration({ duration: 2 }), '2.0s');
assert.equal(formatDuration({}), '--');

const frontendHtml = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert.doesNotMatch(frontendHtml, /overallConf|statLowConf|segment-confidence|uncertain-word/);
assert.doesNotMatch(frontendHtml, /formatConfidence|confidenceTier|summarizeConfidence|speaker_confidence/);

console.log('presentation tests passed');
