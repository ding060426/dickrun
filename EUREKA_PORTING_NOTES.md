# Eureka114514 Porting Notes

Date: 2026-07-16

## Source

- Current HUIWU source: `D:/HUIWU/`
- Target clone: `D:/HUIWU_EUREKA_PORT/`
- Eureka source branch: `https://github.com/ding060426/dickrun/tree/agent/realtime-mic-visualizer`
- Local Eureka clone: `C:/Users/dmm17/AppData/Local/Temp/huiwu-eureka/`

## Policy

- Original `D:/HUIWU/` is not modified.
- Eureka logic is the reference for fields, diarization, logic validation wiring, VAD + endpoint detection, ASR transcription, and hotword correction.
- Current HUIWU features retained where possible: local memory, DTP2, RNNoise, upload progress, logs, frontend, Demo fallback.

## Migrated Files

This section is updated as tasks complete.

## Adaptations

This section is updated as integration changes are made.

## Verification

This section is updated with commands and results.

### Task 1 verification (2026-07-16)

- Removed copied log files from destination clone only: `rm -f /d/HUIWU_EUREKA_PORT/backend/logs/*.log` -> completed with no output.
- Critical file check: `start.py`, `backend/main.py`, `backend/xasr/asr_engine.py`, and `frontend/index.html` all exist in `D:/HUIWU_EUREKA_PORT/`.
- Log exclusion check: no files matching `D:/HUIWU_EUREKA_PORT/backend/logs/*.log` exist.
- Notes check: this section records Task 1 verification results.

## Remaining Risks

This section is updated with missing models, missing optional dependencies, or runtime limitations.

### Foundation contracts and utilities

Copied from Eureka:

- `backend/xasr/contracts.py`
- `backend/xasr/config.py`
- `backend/xasr/file_vad.py`
- `backend/audio_buffer.py`
- `backend/build_info.py`

Verification:

- Contract/profile test: run after Task 2.

### Hotword correction

Copied from Eureka:

- `backend/xasr/hotwords.py`
- `backend/xasr/hotword_config.py`
- `backend/modules/hotword_processor.py`

The cloned program will use Eureka `HotwordProcessor` for canonical alias and fuzzy-pinyin hotword correction.

Verification:

- Hotword processor test: `cd /d/HUIWU_EUREKA_PORT/backend && python -m pytest tests/test_eureka_port_hotword_processor.py -v` -> 2 passed, 1 warning (jieba/pkg_resources deprecation).

### Diarization

Copied from Eureka:

- `backend/diarization/`

The diarization backend reports unavailable instead of crashing when sherpa-onnx or local pyannote/3D-Speaker models are missing.

### Runtime, live audio, upload storage

Copied from Eureka:

- `backend/xasr/runtime_config.py`
- `backend/xasr/engine_pool.py`
- `backend/xasr/live_audio.py`
- `backend/upload_storage.py`

Dependencies were merged into `backend/requirements.txt` without removing existing HUIWU requirements.

Adaptation: `backend/xasr/live_audio.py` imports `.recording`, which was not listed in the original Task 5 copy set. Copied the minimal required Eureka module `backend/xasr/recording.py` after the first import check failed with `ModuleNotFoundError: No module named 'xasr.recording'`.

Verification:

- Runtime/live/upload import check: `cd /d/HUIWU_EUREKA_PORT/backend && python - <<'PY' ... PY` -> `OK: runtime/live/upload imports pass` after copying `backend/xasr/recording.py`.
- Dependency exact-count check: `sherpa-onnx`, `pypinyin`, `soundfile`, `pydub`, `librosa`, and `pytest` each appear exactly once as unpinned Task 5 declarations.

### ASR engine

Started from Eureka `backend/xasr/asr_engine.py` and adapted it for current HUIWU compatibility:

- accepts `cancel_event` and `denoise_enabled` on `process_file`;
- keeps Eureka continuous timeline decode and isolated fallback;
- uses Eureka `ASRResult` contract;
- uses Eureka `HotwordProcessor`;
- preserves current HUIWU ASR optimizer report compatibility.

Task 6 also copied Eureka `backend/xasr/sherpa_streaming_infer.py` into the protected clone after the import check exposed that Eureka `asr_engine.py` depends on `SherpaRecognizerRuntime`, which was absent from the older cloned runtime wrapper.

Verification:

- ASR engine compile: `cd /d/HUIWU_EUREKA_PORT/backend && python -m py_compile xasr/asr_engine.py` -> passed with no output.
- ASR engine import/compatibility check: `cd /d/HUIWU_EUREKA_PORT/backend && python - <<'PY' ... PY` -> `OK: integrated XASREngine imports and compatibility methods exist` (expected missing-model warnings only).
- Contract regression test: `cd /d/HUIWU_EUREKA_PORT/backend && python -m pytest tests/test_eureka_port_contracts.py -q` -> 4 passed, 1 warning (jieba/pkg_resources deprecation).

### Backend main startup wiring

Integrated Eureka runtime/hotword config, engine pool, and diarization pipeline into cloned `backend/main.py` with fail-open behavior. Current HUIWU local memory, DTP2, RNNoise, upload progress, logs, frontend, and Demo fallback remain in the cloned main program.

### Upload processing

Upload recognition now routes through `final_xasr_engine` and Eureka `OfflineMeetingPipeline` when available. If diarization is unavailable or fails, upload recognition falls back to ASR-only and includes diarization metadata explaining the fallback.

Verification:

- Task 8 upload helper tests: `cd /d/HUIWU_EUREKA_PORT/backend && python -m pytest tests/test_eureka_port_main_imports.py -q` -> initially failed as expected before implementation (missing Eureka fields/helper), then passed after implementation; final review-fix run -> 9 passed, 1 warning (jieba/pkg_resources deprecation).
- Task 8 compile: `cd /d/HUIWU_EUREKA_PORT/backend && python -m py_compile main.py diarization/pipeline.py` -> passed with no output.
- Task 8 import check: `cd /d/HUIWU_EUREKA_PORT/backend && python - <<'PY' ... PY` -> `OK: upload pipeline helper is importable` (normal startup logs).
- Task 8 regression set: `cd /d/HUIWU_EUREKA_PORT/backend && python -m pytest tests/test_eureka_port_contracts.py tests/test_eureka_port_main_imports.py -q` -> 13 passed, 1 warning (jieba/pkg_resources deprecation).

### Realtime live path

The cloned `/ws/live` keeps current HUIWU DTP2 and RNNoise compatibility. Live recognition uses `xasr_engine.fork_session()` when Eureka engine pool is available, and live responses include Eureka speaker/word/postprocess fields where present.

Verification:

- Task 9 focused tests: `cd /d/HUIWU_EUREKA_PORT/backend && python -m pytest tests/test_eureka_port_main_imports.py::test_live_asr_engine_forks_from_pool_engine_when_available tests/test_eureka_port_main_imports.py::test_live_asr_engine_constructs_compat_engine_without_fork tests/test_eureka_port_main_imports.py::test_live_realtime_response_includes_eureka_fields -q` -> 3 passed, 1 warning (jieba/pkg_resources deprecation).
- Task 9 compile, DTP2 smoke, and live regression tests: `cd /d/HUIWU_EUREKA_PORT/backend && python -m py_compile main.py modules/realtime_session.py modules/dtp2_protocol.py xasr/live_audio.py && python - <<'PY' ... PY && python -m pytest tests/test_eureka_port_main_imports.py -q` -> `OK: DTP2 parser still works`; 14 passed, 1 warning (jieba/pkg_resources deprecation).

## Verification Results

- Core compile: PASS - `cd /d/HUIWU_EUREKA_PORT/backend && python -m py_compile main.py xasr/asr_engine.py xasr/contracts.py xasr/config.py xasr/file_vad.py xasr/hotwords.py xasr/hotword_config.py xasr/runtime_config.py xasr/engine_pool.py xasr/live_audio.py modules/hotword_processor.py modules/realtime_session.py` completed with no output.
- Diarization compile: PASS - compiled `diarization/alignment.py`, `chunked_backend.py`, `contracts.py`, `pipeline.py`, `registry.py`, `sherpa_backend.py`, `smoothing.py`, and `__init__.py`; printed `OK: diarization package compiles`.
- Contract tests: PASS - `python -m pytest tests/test_eureka_port_contracts.py tests/test_eureka_port_hotword_processor.py tests/test_eureka_port_main_imports.py -v` collected 24 items and passed all 24 with 1 warning (`jieba`/`pkg_resources` deprecation).
- Hotword tests: PASS - included in the same pytest run; `tests/test_eureka_port_hotword_processor.py` passed 2 tests.
- Main import: PASS - backend import printed `app_title=DiTing - Smart Meeting Speech Cognitive System`, `has_memory_store=True`, `has_eureka_pipeline=True`, and `OK: final backend import check complete`.
- DTP2 parser smoke test: PASS - `parse_binary_message` accepted a DTP2 PCM frame (`kind=audio protocol=dtp2 seq=7 samples=4`), JSON `dtp2.audio` control parsing remained available, and the command printed `OK: DTP2 parser smoke test complete`. An initial smoke-script expression using `binary.audio or []` failed because numpy arrays cannot be truth-tested; this was a test-script issue only, rerun with `binary.audio is not None` passed.
- Optional dependency/model availability: PASS WITH LIMITATIONS - `sherpa_onnx`, `pypinyin`, and `pytest` are installed; ASR 160ms encoder/decoder/joiner ONNX files are present; `xasr/models/silero_vad.onnx` and `diarization/models/` are missing.
- Artifact cleanup: PASS - removed generated `backend/logs/*.log`, `backend/.pytest_cache/`, and copied/generated cache directory `backend/xasr/models/.cache/`; follow-up globs found no `logs`, `.cache`, or `.pytest_cache` entries under `backend/`.

## Remaining Risks

- Diarization requires local pyannote segmentation and 3D-Speaker ONNX models under `backend/diarization/models/` or configured environment variables. In this verification environment, `backend/diarization/models/` is missing; integrated tests confirm fail-open behavior when models are absent.
- Silero VAD requires `backend/xasr/models/silero_vad.onnx` or `DITING_SILERO_VAD_PATH`; in this verification environment it is missing, so file/live VAD must use fallback behavior.
- Multiple ASR chunk profiles require matching ONNX model filenames. The 160ms ASR encoder/decoder/joiner models are present, but other profile-specific model files were not found and should be treated as status/fallback cases according to engine pool behavior.
- Optional Python dependencies such as `sherpa-onnx`, `pypinyin`, and `pytest` may need installation from `backend/requirements.txt` on other machines. They are currently available in this environment.
- Importing `backend/main.py` initializes the DiTing logging system and creates log files; Task 10 removed the generated logs after verification to keep the clone free of copied/generated log artifacts.
