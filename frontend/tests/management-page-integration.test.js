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
assert.match(html, /buildRecordPayload/);
assert.match(html, /id="btnNewRecord"/);
assert.match(html, /id="btnSaveRecord"/);
assert.match(html, /id="recordsPanel"/);
assert.match(html, /\/api\/records/);
assert.match(html, /id="micOrbPanel"/);
assert.match(html, /setMicVisualizerState\('recording'\)/);
assert.match(html, /event\.data\?\.type === 'audio\.level'/);
assert.match(html, /await ensureAppSettingsLoaded\(\)/);
assert.match(html, /buildMediaConstraints\(microphoneSettings\)/);
assert.match(html, /profile: microphoneSettings\.live_profile \|\| 'meeting'/);
assert.doesNotMatch(html, /profile: 'meeting'/);
assert.match(html, /id="asrProviderXasr"/);
assert.match(html, /id="asrProviderQwen3"/);
assert.match(html, /id="qwen3ModelPath"/);
assert.match(html, /asr_provider: selectedAsrProvider/);
assert.match(html, /setAsrProviderUI\(recognition\.asr_provider/);
assert.match(html, /--settings-accent: #7c3aed/);
assert.match(html, /\.asr-provider-card\.active/);
assert.match(html, /remove\.textContent = '×'/);
assert.match(html, /settings\.hotwords\.enable_word/);
assert.match(html, /settings\.hotwords\.delete_word/);
  assert.doesNotMatch(html, /脳|鍚敤|鐑瘝|鏉冮噸|鍒犻櫎/);
  assert.match(html, /input\[type="range"\].*accent-color:\s*var\(--settings-accent\)/s);
  assert.doesNotMatch(html, /rgba\(15, 17, 23, 0\.48\)/);
assert.match(html, /` · \$\{t\('settings\.model_not_deployed'\)\}`/);
assert.doesNotMatch(html, /` 路 \$\{t\('settings\.model_not_deployed'\)\}`/);
assert.doesNotMatch(html, /ws:\/\/localhost:8765/);
assert.doesNotMatch(html, /id="btnStart"/);
assert.doesNotMatch(html, /function startDemo\(/);
assert.doesNotMatch(html, /getEmbeddedDemoData/);
assert.doesNotMatch(html, /\/api\/meeting\/demo/);
assert.doesNotMatch(html, /demo_mode/);
assert.match(html, /id="reservationParticipants"/);
assert.match(html, /participant_user_ids:/);
assert.match(html, /function editReservation\(/);
assert.match(html, /method: 'PUT'/);
assert.match(html, /organizer\?\.display_name/);
assert.match(html, /participants \|\| \[\]/);
assert.match(html, /can_manage/);
assert.match(html, /new Date\(startTime\)\.toISOString\(\)/);
assert.match(html, /id="accountMenuButton"/);
assert.match(html, /async function saveCurrentProfile\(/);
assert.match(html, /id="profileAvatarInput"/);
assert.match(html, /async function switchUser\(/);
assert.match(html, /await logout\(\)/);
assert.match(html, /apiFetch\('\/api\/auth\/me'/);
assert.match(html, /canvas\.toDataURL\('image\/jpeg'/);
assert.ok(
  html.indexOf('id="profileOverlay"') < html.indexOf('id="mainApp"'),
  'profile editor must stay outside the hidden main app container',
);

const moduleScript = html.match(/<script type="module">([\s\S]*?)<\/script>/);
assert.ok(moduleScript, 'inline module script should exist');
new Function(moduleScript[1]);

console.log('management page integration tests passed');
