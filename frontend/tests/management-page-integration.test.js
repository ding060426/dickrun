const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');


const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert.match(html, /data-module="meeting">会议转写/);
assert.match(html, /buildUploadUrl\(API_BASE/);
assert.match(html, /enableDiarization/);
assert.match(html, /DiTingLiveProtocol\.encodePcmFrame/);
assert.match(html, /msg\.type === 'final_transcript'/);
assert.match(html, /buildAnalysisPayload/);
assert.doesNotMatch(html, /ws:\/\/localhost:8765/);

const moduleScript = html.match(/<script type="module">([\s\S]*?)<\/script>/);
assert.ok(moduleScript, 'inline module script should exist');
new Function(moduleScript[1]);

console.log('management page integration tests passed');
