# HUIWU Integration Changelog

## 2026-07-16

### Added

- Created HUIWU integration copy at `D:\HUIWU` from current `D:\diting4.5` baseline.
- Added optional DTP2 realtime audio protocol support.
- Added PCM helper utilities.
- Added optional RNNoise ctypes wrapper with fail-open behavior.
- Added lightweight realtime VAD/session orchestration.
- Added frontend helper files from `agent/realtime-mic-visualizer`:
  - `audio-worklet.js`
  - `live-protocol.js`
  - `mic-level.js`
  - `app-settings.js`
- Added AudioWorklet + DTP2 frontend microphone path with fallback to the previous JSON/base64 Float32 protocol.
- Added HUIWU handoff document.

### Changed

- `/ws/live` now advertises supported protocols:
  - `json.float32`
  - `dtp2.pcm_s16le`
- `/ws/live` now accepts both text JSON and binary DTP2 frames.
- `/api/health` now reports:
  - `service: HUIWU / DiTing v2.0`
  - `huiwu_realtime`
  - `rnnoise` status.
- `ASROptimizer` now includes `denoise` report fields.

### Preserved

- Upload recognition flow.
- Upload progress WebSocket.
- Local memory/history APIs.
- Meeting summary API and fallback behavior.
- Existing frontend history panel and result display.
- Old realtime mic protocol fallback.

### Validation

- Python syntax compile passed.
- Frontend module syntax check passed.
- `python start.py` launched backend/frontend.
- `GET /api/health` returned 200.
- `GET /api/memory/history` returned 200.
- Frontend `http://localhost:3000` returned 200.
- DTP2 WebSocket smoke test passed.
- Legacy JSON WebSocket smoke test passed.
