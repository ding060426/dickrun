"""
谛听 (DiTing) - Smart Meeting Speech Cognitive System
Backend Server v4.5 (2026-07-15)
==========================================================================
Changelog:
  - Fixed ASR engine: per-utterance processing with VAD+endpoint detection
  - Added centralized logging (console + file + ring buffer)
  - Added Eval_Ali dataset integration endpoints
  - Added log streaming endpoint for frontend debugging
"""

import asyncio
import json
import os
import sys
import time
import uuid
import struct
import base64
import tempfile
import threading
import concurrent.futures
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Paths ───────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# ── Logging ─────────────────────────────────────────────────────
from utils.logger import init_logging, get_logger, get_recent_logs as get_recent_log_entries, log_buffer

init_logging(console_level="INFO", file_level="DEBUG")
logger = get_logger("main")

# ── X-ASR ───────────────────────────────────────────────────────
try:
    from xasr.asr_engine import XASREngine, ASRResult
    HAS_XASR = True
    logger.info("X-ASR module loaded")
except ImportError as e:
    logger.warning(f"X-ASR module load failed: {e}")
    logger.warning("Will use demo mode only")
    HAS_XASR = False

# ── Eval_Ali ────────────────────────────────────────────────────
try:
    import eval_ali_integration as eval_ali
    HAS_EVAL = True
    logger.info("Eval_Ali integration loaded")
except ImportError as e:
    logger.warning(f"Eval_Ali integration not available: {e}")
    HAS_EVAL = False

# ── Text Post-Processor ───────────────────────────────────────
try:
    from modules.text_post_processor import postprocess_text
    HAS_POSTPROCESSOR = True
    logger.info("Text post-processor loaded")
except ImportError as e:
    logger.warning(f"Text post-processor not available: {e}")
    HAS_POSTPROCESSOR = False

# ── VAD Manager ───────────────────────────────────────────────
try:
    from modules.vad_manager import segment_audio as vad_segment_audio, get_vad_info
    HAS_VAD_MANAGER = True
    logger.info("VAD manager loaded (FireRedVAD → Silero → Energy)")
except ImportError as e:
    logger.warning(f"VAD manager not available: {e}")
    HAS_VAD_MANAGER = False

# ── Action Extractor ─────────────────────────────────────────
try:
    from modules.action_extractor import extract_action_items, ActionExtractor
    HAS_ACTION_EXTRACTOR = True
    logger.info("Action extractor loaded")
except ImportError as e:
    logger.warning(f"Action extractor not available: {e}")
    HAS_ACTION_EXTRACTOR = False

# ===========================================================================
# Lifespan: Load X-ASR in background
# ===========================================================================

xasr_engine: Optional[XASREngine] = None
xasr_loading: bool = False

def _load_xasr_engine():
    """Load X-ASR engine in background thread (fire-and-forget)."""
    global xasr_engine, xasr_loading
    if not HAS_XASR:
        logger.info("X-ASR not available, demo mode only")
        return
    xasr_loading = True
    logger.info("Loading X-ASR model in background thread (~300MB)...")
    try:
        engine = XASREngine(
            hotwords=DEMO_MEETING["hotwords"],
            enable_logic_validation=True,
            enable_hotword_correction=True,
            enable_uncertainty=True,
            enable_endpoint_detection=True,
            provider="cpu",
            num_threads=2,
        )
        xasr_engine = engine
        if engine.is_model_available:
            logger.info("X-ASR model loaded successfully!")
        else:
            logger.warning("X-ASR model not found, using demo mode")
    except Exception as e:
        logger.error(f"X-ASR init failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        xasr_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: fire-and-forget model loading."""
    logger.info("Starting DiTing backend server...")
    threading.Thread(target=_load_xasr_engine, daemon=True).start()
    yield
    logger.info("Shutting down DiTing backend...")

app = FastAPI(
    title="DiTing - Smart Meeting Speech Cognitive System",
    version="4.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Demo Data (unchanged - used for demo mode)
# ===========================================================================

DEMO_MEETING = {
    "title": "2024Q3 Product Review",
    "date": "2024-07-14 14:00",
    "duration": "04:30",
    "participants": [
        {"id": "SPK_1", "name": "Zhang San", "role": "PM", "color": "#4A90D9"},
        {"id": "SPK_2", "name": "Li Si", "role": "Tech Lead", "color": "#E8743C"},
        {"id": "SPK_3", "name": "Wang Wu", "role": "Ops", "color": "#50B86C"},
    ],
    "hotwords": [
        "BERT", "Transformer", "A/B Test", "Conversion", "Multimodal",
        "Full Users", "New Users", "Q3", "OKR", "Fine-tuning",
        "Attention", "Recommendation", "Review", "Budget", "User Base"
    ],
    "segments": [
        {
            "start": 0.0, "end": 5.5, "speaker": "SPK_1", "snr_db": 28, "quality": "high",
            "raw_text": "Today we review Q3 conversion data, we fine-tuned BERT-based recommendation model, conversion went from 10% to 15%.",
            "display_text": "Today we review Q3 conversion data, we fine-tuned BERT-based recommendation model, conversion went from 10% to 15%.",
            "corrections": [{"pos": 21, "original": "bat", "corrected": "BERT", "method": "pinyin_match"}],
            "terms": ["Q3", "BERT", "Recommendation", "Fine-tuning", "Conversion"],
            "data_points": [{"value": "10%", "type": "baseline"}, {"value": "15%", "type": "result"}],
            "logic_flags": []
        },
        {
            "start": 6.0, "end": 12.0, "speaker": "SPK_2", "snr_db": 26, "quality": "high",
            "raw_text": "Wait, my backend data shows conversion is only 8%, that doesn't match your numbers.",
            "display_text": "Wait, my backend data shows conversion is only 8%, that doesn't match your numbers.",
            "corrections": [], "terms": ["Conversion"],
            "data_points": [{"value": "8%", "type": "conflict"}],
            "logic_flags": [{"type": "data_conflict", "severity": "warning",
                             "message": "Data conflict: Zhang San claims 15% vs Li Si claims 8%",
                             "conflict_with": {"speaker": "SPK_1", "time": "00:02", "claim": "15%"},
                             "resolution": "Pending - may be different metric definitions"}]
        },
    ],
    "summary": {
        "topics": ["Q3 conversion data alignment - resolved (Full Users vs New Users)"],
        "todos": [{"assignee": "Zhang San", "task": "Unify conversion metrics", "deadline": "by 07/21"}],
        "low_confidence_spots": [],
        "stats": {"total_segments": 2, "logic_flags": 1, "low_confidence": 0, "resolved": 0, "overall_confidence": 0.85}
    }
}

# ===========================================================================
# WebSocket Manager
# ===========================================================================

class ConnectionManager:
    """Track active WebSocket connections."""
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def send_json(self, ws: WebSocket, data: dict):
        try:
            await ws.send_json(data)
        except Exception:
            self.disconnect(ws)

manager = ConnectionManager()
upload_sessions: dict = {}
upload_cancel_events: dict = {}
upload_tasks: dict = {}
upload_tasks_lock = threading.RLock()
upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
UPLOAD_TASK_TTL_SEC = 3600
UPLOAD_TERMINAL_STATUSES = {"completed", "error", "cancelled", "demo_mode"}


def _now_ts() -> float:
    return time.time()


def _cleanup_upload_tasks():
    """Remove old terminal upload tasks from the in-memory task table."""
    now = _now_ts()
    with upload_tasks_lock:
        stale_ids = [
            file_id for file_id, task in upload_tasks.items()
            if task.get("status") in UPLOAD_TERMINAL_STATUSES
            and now - float(task.get("updated_at", now)) > UPLOAD_TASK_TTL_SEC
        ]
        for file_id in stale_ids:
            upload_tasks.pop(file_id, None)


def _new_upload_task(file_id: str, filename: str = None, size_mb: float = 0.0, engine: str = "") -> dict:
    """Create or refresh an in-memory upload task."""
    _cleanup_upload_tasks()
    now = _now_ts()
    with upload_tasks_lock:
        task = upload_tasks.get(file_id) or {
            "file_id": file_id,
            "created_at": now,
            "segments": [],
            "summary": None,
            "domain": None,
            "hotwords": [],
            "speaker_stats": {},
            "error": None,
            "cancel_requested": False,
        }
        task.update({
            "filename": filename or task.get("filename"),
            "status": "processing",
            "size_mb": round(float(size_mb or 0.0), 2),
            "engine": engine or task.get("engine", ""),
            "updated_at": now,
            "progress_stage": "upload_saved",
            "progress_fraction": 0.0,
            "segments_count": len(task.get("segments", [])),
        })
        upload_tasks[file_id] = task
        return task


def _update_upload_task(file_id: str, **fields) -> dict:
    now = _now_ts()
    with upload_tasks_lock:
        task = upload_tasks.get(file_id) or {
            "file_id": file_id,
            "created_at": now,
            "segments": [],
            "summary": None,
            "domain": None,
            "hotwords": [],
            "speaker_stats": {},
            "error": None,
            "cancel_requested": False,
        }
        task.update(fields)
        task["updated_at"] = now
        upload_tasks[file_id] = task
        return task


def _append_upload_segment(file_id: str, segment: dict, segment_index: int = None, total_estimated: int = None) -> dict:
    with upload_tasks_lock:
        task = upload_tasks.get(file_id) or _update_upload_task(file_id)
        segments = list(task.get("segments", []))
        if segment_index is None:
            segments.append(segment)
        else:
            while len(segments) <= segment_index:
                segments.append(None)
            segments[segment_index] = segment
        compact_segments = [s for s in segments if s]
        task["segments"] = compact_segments
        task["segments_count"] = len(compact_segments)
        if total_estimated is not None:
            task["total_estimated"] = total_estimated
        task["updated_at"] = _now_ts()
        upload_tasks[file_id] = task
        return task


def _get_upload_task(file_id: str) -> Optional[dict]:
    with upload_tasks_lock:
        task = upload_tasks.get(file_id)
        return dict(task) if task else None


def _serialize_upload_task(task: dict, include_segments: bool = True) -> dict:
    data = dict(task or {})
    if not include_segments:
        data.pop("segments", None)
    return _sanitize(data)


async def _broadcast_upload(file_id: str, msg: dict):
    queues = list(upload_sessions.get(file_id, set()) or [])
    for queue in queues:
        try:
            await queue.put(msg)
        except Exception:
            pass


def _broadcast_upload_threadsafe(file_id: str, msg: dict, loop):
    try:
        asyncio.run_coroutine_threadsafe(_broadcast_upload(file_id, msg), loop)
    except Exception as e:
        logger.warning(f"Upload broadcast failed for {file_id[:8]}: {e}")

# ===========================================================================
# API Routes
# ===========================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "DiTing v4.5",
        "xasr_available": HAS_XASR and xasr_engine is not None and xasr_engine.is_model_available,
        "xasr_loading": xasr_loading,
        "eval_available": HAS_EVAL,
        "timestamp": time.time(),
    }


@app.get("/api/xasr/status")
async def get_xasr_status():
    """Get X-ASR engine status with detail."""
    if not HAS_XASR:
        return {"available": False, "reason": "X-ASR module not installed", "loading": False}
    if xasr_loading:
        return {"available": True, "reason": "Loading model...", "model_available": False, "loading": True}
    if xasr_engine is None:
        return {"available": False, "reason": "Engine init failed", "model_available": False, "loading": False}

    return {
        "available": True,
        "model_available": xasr_engine.is_model_available,
        "model_dir": xasr_engine.model_dir,
        "endpoint_detection": xasr_engine.enable_endpoint_detection,
        "hotwords_count": len(xasr_engine.hotword_corrector.hotwords) if xasr_engine.hotword_corrector else 0,
        "features": {
            "logic_validation": xasr_engine.enable_logic_validation,
            "hotword_correction": xasr_engine.enable_hotword_correction,
            "uncertainty_estimation": xasr_engine.enable_uncertainty,
        },
        "loading": False,
    }


@app.get("/api/meeting/demo")
async def get_demo_meeting():
    """Return demo meeting data."""
    return DEMO_MEETING


@app.get("/api/hotwords")
async def get_hotwords():
    """Get current hotword list."""
    engine_words = list(xasr_engine.hotword_corrector.hotwords) if (
        xasr_engine and xasr_engine.hotword_corrector
    ) else []
    return {
        "hotwords": list(set(DEMO_MEETING["hotwords"] + engine_words)),
        "count": len(DEMO_MEETING["hotwords"]) + len(engine_words),
    }


@app.post("/api/hotwords")
async def add_hotwords(data: dict):
    """Add custom hotwords."""
    new_words = data.get("words", [])
    if xasr_engine:
        xasr_engine.add_hotwords(new_words)
    return {"added": new_words, "ok": True}


# ===========================================================================
# Log endpoints
# ===========================================================================

@app.get("/api/logs/recent")
async def api_get_recent_logs(n: int = Query(50, ge=1, le=500)):
    """Get recent log entries (for frontend debug panel)."""
    return {"logs": get_recent_log_entries(n), "count": len(log_buffer)}


@app.get("/api/logs/download")
async def download_logs():
    """Download the full log file."""
    log_file = os.path.join(BACKEND_DIR, "logs", "diting.log")
    if os.path.exists(log_file):
        return FileResponse(log_file, filename="diting.log", media_type="text/plain")
    return {"error": "Log file not found"}


# ===========================================================================
# Eval_Ali endpoints
# ===========================================================================

@app.get("/api/eval/status")
async def get_eval_status():
    """Get Eval_Ali dataset status."""
    if not HAS_EVAL:
        return {"available": False, "reason": "eval_ali_integration not loaded"}
    status = eval_ali.check_dataset_status()
    return {
        "available": True,
        "eval_ali_exists": status['eval_ali']['exists'],
        "meetings": status['eval_ali']['scan']['total_meetings'],
        "audio_hours_est": round(status['eval_ali']['scan']['total_audio_hours'], 1),
        "mug_exists": status['mug']['exists'],
    }


@app.get("/api/eval/hotwords")
async def get_eval_hotwords(max_count: int = Query(50, ge=10, le=500)):
    """Extract domain hotwords from AliMeeting4MUG dataset."""
    if not HAS_EVAL:
        return {"available": False, "hotwords": []}
    try:
        hotwords = eval_ali.extract_hotwords_from_mug(max_hotwords=max_count)
        return {"available": True, "hotwords": hotwords, "count": len(hotwords)}
    except Exception as e:
        return {"available": False, "error": str(e), "hotwords": []}


@app.get("/api/eval/meeting/{meeting_id}")
async def get_eval_meeting(meeting_id: str):
    """Get details for a specific Eval_Ali meeting (TextGrid transcript)."""
    if not HAS_EVAL:
        return {"error": "Eval_Ali integration not available"}
    scan = eval_ali.scan_eval_dataset(use_far=True, use_near=False)
    mtg = scan['meetings'].get(meeting_id)
    if not mtg:
        return {"error": f"Meeting {meeting_id} not found", "available": list(scan['meetings'].keys())}
    return {
        "meeting_id": meeting_id,
        "far_wavs": mtg.get('far_wavs', []),
        "transcript_sample": mtg.get('far_transcript', [])[:20],
        "total_utterances": len(mtg.get('far_transcript', [])),
    }


# ===========================================================================
# Speaker endpoints (v3.1)
# ===========================================================================

@app.get("/api/speakers")
async def get_speakers():
    """List all enrolled speakers."""
    if not xasr_engine or not xasr_engine.speaker_identifier:
        return {"speakers": [], "count": 0, "available": False}
    speakers = xasr_engine.speaker_identifier.list_speakers()
    return {
        "speakers": speakers,
        "count": len(speakers),
        "available": True,
    }


@app.post("/api/speakers/enroll")
async def enroll_speaker(data: dict):
    """Enroll a new speaker (requires audio upload via multipart)."""
    return {
        "ok": False,
        "message": "Use POST /api/audio/upload with enroll_speaker=true&speaker_name=NAME",
        "alternate": "GET /api/speakers/enroll_from_eval to auto-enroll from Eval_Ali dataset",
    }


@app.get("/api/speakers/enroll_from_eval")
async def enroll_from_eval():
    """Auto-enroll all speakers from Eval_Ali near-field dataset (25 speakers)."""
    if not xasr_engine or not xasr_engine.speaker_identifier:
        # Try to init speaker identifier on-demand
        if xasr_engine:
            try:
                from modules.speaker_diarization import SpeakerIdentifier
                xasr_engine.speaker_identifier = SpeakerIdentifier(embed_dim=256)
            except Exception as e:
                return {"ok": False, "count": 0, "error": str(e)}
        else:
            return {"ok": False, "count": 0, "error": "X-ASR engine not loaded"}

    count = xasr_engine.speaker_identifier.enroll_from_eval_ali()
    return {
        "ok": count > 0,
        "count": count,
        "speakers": xasr_engine.speaker_identifier.list_speakers(),
    }


# ===========================================================================
# Domain & Cognitive endpoints (v3.1)
# ===========================================================================

@app.get("/api/domain/taxonomy")
async def get_domain_taxonomy():
    """Get the full domain taxonomy."""
    from modules.domain_taxonomy import DOMAIN_TAXONOMY, get_all_domains
    domains = []
    for domain_name in get_all_domains():
        info = DOMAIN_TAXONOMY[domain_name]
        domains.append({
            "name": domain_name,
            "description": info.get("description", ""),
            "sub_domains": info.get("sub_domains", []),
            "keyword_count": len(info.get("keywords", [])),
        })
    return {"domains": domains, "count": len(domains)}


@app.post("/api/domain/infer")
async def infer_domain(data: dict):
    """Infer meeting domain from a list of hotwords/terms."""
    hotwords = data.get("hotwords", data.get("terms", []))
    if not hotwords:
        return {"error": "No hotwords provided", "domain": None}

    from modules.cognitive_engine import DomainInferrer
    inferrer = DomainInferrer()
    result = inferrer.infer(hotwords)
    return result


@app.get("/api/cognitive/status")
async def get_cognitive_status():
    """Get cognitive enhancement system status."""
    from modules.llm_client import get_llm_client
    llm = get_llm_client()

    return {
        "llm_available": llm.is_available,
        "llm_provider": type(llm).__name__,
        "speaker_identification": bool(
            xasr_engine and xasr_engine.speaker_identifier
            and xasr_engine.speaker_identifier.has_enrolled()
        ),
        "enrolled_speakers": len(
            xasr_engine.speaker_identifier.enrolled
        ) if (xasr_engine and xasr_engine.speaker_identifier) else 0,
        "domain_taxonomy": len(DOMAIN_TAXONOMY) if 'DOMAIN_TAXONOMY' in dir() else 5,
    }


@app.get("/api/vad/status")
async def get_vad_status():
    """Get VAD (Voice Activity Detection) system status."""
    if HAS_VAD_MANAGER:
        info = get_vad_info()
        return {
            "available": True,
            **info,
            "active_vad": info.get("active_vad") or "auto",
            "priority_label": "FireRedVAD → Silero → Energy",
        }
    return {"available": False, "active_vad": "energy_fallback"}


@app.post("/api/action-items")
async def extract_actions(request: Request):
    """
    从会议 segments 中提取行动项。

    Request body:
        {
            "segments": [
                {"speaker": "张三", "text": "...", "start": 0.0, "end": 5.0},
                ...
            ]
        }

    Returns:
        {
            "action_items": [
                {
                    "task": "整理测试报告",
                    "assignee": "张三",
                    "deadline": "周五前",
                    "priority": "high",
                    "source_text": "张三负责整理测试报告，周五前完成",
                    "speaker": "主持人",
                    "segment_index": 2
                },
                ...
            ],
            "count": 2,
            "extractor": "rules"  # or "llm"
        }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body"}
        )

    segments = body.get("segments", [])
    if not segments:
        return {"action_items": [], "count": 0, "extractor": "none"}

    if not HAS_ACTION_EXTRACTOR:
        return JSONResponse(
            status_code=503,
            content={"error": "Action extractor not available", "action_items": [], "count": 0}
        )

    actions = await asyncio.to_thread(extract_action_items, segments)

    return {
        "action_items": actions,
        "count": len(actions),
        "extractor": "llm" if actions and any(a.get("source_text") for a in actions) else "rules",
    }


# ===========================================================================
# Audio Upload (with real-time WebSocket progress)
# ===========================================================================

def _sanitize(val):
    """Convert numpy types to JSON-safe Python types."""
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize(v) for v in val]
    return val


def _numpy_to_wav_base64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Convert a float32 numpy audio array to a base64-encoded WAV string.

    Args:
        audio: 1-D numpy float32 array in range [-1, 1]
        sample_rate: Sample rate in Hz (default: 16000)

    Returns:
        Base64-encoded WAV file as an ASCII string
    """
    audio = np.asarray(audio, dtype=np.float32)
    # Normalize to [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    n_samples = len(audio_int16)
    byte_rate = sample_rate * 2       # 16-bit mono = 2 bytes/sample
    block_align = 2                   # 1 channel × 2 bytes
    data_size = n_samples * 2

    # Canonical 44-byte WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,               # ChunkSize
        b'WAVE',
        b'fmt ',                       # Subchunk1ID
        16,                            # Subchunk1Size (PCM = 16)
        1,                             # AudioFormat (PCM = 1)
        1,                             # NumChannels (mono)
        sample_rate,                   # SampleRate
        byte_rate,                     # ByteRate
        block_align,                   # BlockAlign
        16,                            # BitsPerSample
        b'data',                       # Subchunk2ID
        data_size,                     # Subchunk2Size
    )

    wav_bytes = header + audio_int16.tobytes()
    return base64.b64encode(wav_bytes).decode('ascii')


def _result_to_dict(result: ASRResult, index: int) -> dict:
    """Convert ASRResult to JSON-safe dict."""
    return _sanitize({
        "index": index,
        "text": result.text,
        "raw_text": result.raw_text,
        "speaker_id": getattr(result, 'speaker_id', 'unknown'),
        "start_sec": result.start_sec,
        "end_sec": result.end_sec,
        "asr_confidence": round(float(result.asr_confidence), 3),
        "snr_db": round(float(result.snr_db), 1),
        "rt60": round(float(result.rt60), 3),
        "quality_score": round(float(result.quality_score), 3),
        "quality_label": result.quality_label,
        "corrections": result.corrections,
        "logic_flags": result.logic_flags,
        "terms": result.terms,
        "data_points": result.data_points,
        "uncertain_spans": result.uncertain_spans,
        "uncertainty": result.uncertainty,
    })


@app.post("/api/audio/upload/{file_id}/cancel")
async def cancel_upload(file_id: str):
    """Cancel an in-progress upload recognition task."""
    task = _get_upload_task(file_id)
    if task and task.get("status") in UPLOAD_TERMINAL_STATUSES:
        return {"ok": True, "file_id": file_id, "cancel_requested": False, "status": task.get("status")}

    event = upload_cancel_events.get(file_id)
    if event:
        event.set()
    _update_upload_task(file_id, status="cancelling", cancel_requested=True)
    await _broadcast_upload(file_id, {
        "type": "cancelled",
        "data": {"file_id": file_id, "status": "cancelling", "message": "Upload cancellation requested"},
    })
    logger.info(f"Upload cancellation requested: {file_id[:8]}...")
    return {"ok": True, "file_id": file_id, "cancel_requested": True, "status": "cancelling", "cancelled": bool(event or task)}


@app.get("/api/audio/upload/{file_id}/status")
async def get_upload_status(file_id: str, include_segments: bool = Query(True)):
    """Get current in-memory upload task status."""
    task = _get_upload_task(file_id)
    if not task:
        return {"ok": False, "file_id": file_id, "error": "not_found"}
    return {"ok": True, "task": _serialize_upload_task(task, include_segments=include_segments)}


@app.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...), file_id: str = Query(None)):
    """Upload audio file for X-ASR processing with real-time WebSocket progress."""
    try:
        if not file_id:
            file_id = str(uuid.uuid4())

        # Save uploaded file
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"Upload: {file.filename} ({file_size_mb:.1f}MB) file_id={file_id[:8]}...")

        queue = upload_sessions.get(file_id)
        cancel_event = threading.Event()
        upload_cancel_events[file_id] = cancel_event
        task = _new_upload_task(
            file_id,
            filename=file.filename,
            size_mb=file_size_mb,
            engine="X-ASR (sherpa-onnx zipformer2 v4.5)" if (xasr_engine and xasr_engine.is_model_available) else "Demo (model not loaded)",
        )

        if xasr_engine and xasr_engine.is_model_available:
            if queue:
                # ── WebSocket streaming mode ──────────────────
                await _broadcast_upload(file_id, {
                    "type": "status",
                    "data": {
                        "status": "processing",
                        "message": f"Processing {file.filename} ({file_size_mb:.1f}MB)...",
                        "filename": file.filename,
                        "file_id": file_id,
                        "engine": "X-ASR (sherpa-onnx zipformer2 v4.5)",
                    }
                })

                loop = asyncio.get_event_loop()

                def on_segment(result, idx, total):
                    if cancel_event.is_set():
                        return
                    seg_data = _result_to_dict(result, idx)
                    # Encode raw audio to base64 WAV for frontend waveform
                    audio_b64 = None
                    if result.audio_data is not None and len(result.audio_data) > 0:
                        try:
                            audio_b64 = _numpy_to_wav_base64(result.audio_data, 16000)
                        except Exception as enc_err:
                            logger.warning(f"Audio encoding failed for segment {idx}: {enc_err}")
                    seg_data["audio_wav_base64"] = audio_b64

                    segment_index = idx - 1
                    _append_upload_segment(file_id, seg_data, segment_index=segment_index, total_estimated=total)
                    _broadcast_upload_threadsafe(file_id, {
                        "type": "segment",
                        "data": {
                            "segment": seg_data,
                            "segment_index": segment_index,
                            "total_estimated": total,
                            "cumulative_stats": {"segments_processed": idx},
                        }
                    }, loop)

                def on_progress(stage, fraction):
                    if cancel_event.is_set():
                        return
                    _update_upload_task(file_id, status="processing", progress_stage=stage, progress_fraction=float(fraction))
                    _broadcast_upload_threadsafe(file_id, {
                        "type": "progress",
                        "data": {"stage": stage, "fraction": fraction}
                    }, loop)

                def do_process():
                    try:
                        if cancel_event.is_set():
                            logger.info(f"Processing cancelled before start: {file.filename}")
                            _update_upload_task(file_id, status="cancelled", cancel_requested=True, progress_stage="cancelled")
                            _broadcast_upload_threadsafe(file_id, {
                                "type": "cancelled",
                                "data": {"file_id": file_id, "status": "cancelled", "message": "Upload processing cancelled"},
                            }, loop)
                            return

                        results = xasr_engine.process_file(
                            tmp_path, on_segment=on_segment, on_progress=on_progress, cancel_event=cancel_event
                        )
                        if cancel_event.is_set():
                            logger.info(f"Processing cancelled after ASR: {file.filename}")
                            _update_upload_task(file_id, status="cancelled", cancel_requested=True, progress_stage="cancelled")
                            _broadcast_upload_threadsafe(file_id, {
                                "type": "cancelled",
                                "data": {"file_id": file_id, "status": "cancelled", "message": "Upload processing cancelled"},
                            }, loop)
                            return

                        # Build final segments list with audio
                        final_segments = []
                        for i, r in enumerate(results):
                            seg = _result_to_dict(r, i + 1)
                            audio_b64 = None
                            if r.audio_data is not None and len(r.audio_data) > 0:
                                try:
                                    audio_b64 = _numpy_to_wav_base64(r.audio_data, 16000)
                                except Exception as enc_err:
                                    logger.warning(f"Audio encoding failed for segment {i}: {enc_err}")
                            seg["audio_wav_base64"] = audio_b64
                            final_segments.append(seg)
                        # Build speaker distribution stats
                        speaker_stats = {}
                        for r in results:
                            sid = getattr(r, 'speaker_id', 'unknown')
                            speaker_stats[sid] = speaker_stats.get(sid, 0) + 1

                        # Generate meeting summary (v3.1)
                        meeting_summary = None
                        meeting_domain = None
                        meeting_hotwords = []
                        if xasr_engine and xasr_engine.enable_cognitive:
                            meeting_domain = xasr_engine.get_meeting_domain()
                            meeting_hotwords = xasr_engine.get_meeting_hotwords()
                            try:
                                from modules.cognitive_engine import MeetingSummarizer
                                summarizer = MeetingSummarizer()
                                meeting_data = {
                                    "title": file.filename,
                                    "segments": [
                                        {"speaker": r.speaker_id if hasattr(r, 'speaker_id') else "unknown",
                                         "text": r.text, "terms": getattr(r, 'terms', [])}
                                        for r in results
                                    ],
                                    "domain": meeting_domain,
                                    "participants": [
                                        {"id": name, "name": name, "role": info.get("role", "")}
                                        for name, info in (xasr_engine.speaker_identifier.enrolled.items()
                                                          if xasr_engine.speaker_identifier else {}.items())
                                    ][:10],
                                }
                                meeting_summary = summarizer.summarize(meeting_data)
                                logger.info("Meeting summary generated")
                            except Exception as e:
                                logger.debug(f"Summary generation skipped: {e}")

                        complete_data = {
                            "file_id": file_id,
                            "filename": file.filename,
                            "status": "completed",
                            "engine": "X-ASR (sherpa-onnx zipformer2 v4.5)",
                            "segments_count": len(results),
                            "segments": final_segments,
                            "summary": meeting_summary,
                            "domain": meeting_domain,
                            "hotwords": meeting_hotwords[:20] if meeting_hotwords else [],
                            "speaker_stats": speaker_stats,
                        }
                        _update_upload_task(
                            file_id,
                            status="completed",
                            progress_stage="done",
                            progress_fraction=1.0,
                            segments_count=len(results),
                            segments=final_segments,
                            summary=meeting_summary,
                            domain=meeting_domain,
                            hotwords=complete_data["hotwords"],
                            speaker_stats=speaker_stats,
                        )
                        _broadcast_upload_threadsafe(file_id, {"type": "complete", "data": complete_data}, loop)
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        logger.error(traceback.format_exc())
                        _update_upload_task(file_id, status="error", error=str(e), progress_stage="error")
                        _broadcast_upload_threadsafe(file_id, {"type": "error", "data": {"message": str(e)}}, loop)
                    finally:
                        upload_cancel_events.pop(file_id, None)
                        if xasr_engine.logic_validator:
                            xasr_engine.logic_validator.reset()
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                future = upload_executor.submit(do_process)
                _update_upload_task(file_id, future_id=id(future))

                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "processing",
                    "size_mb": round(file_size_mb, 2),
                    "engine": "X-ASR (sherpa-onnx zipformer2 v4.5)",
                }
            else:
                # ── Synchronous mode (no WebSocket) ────────────
                logger.info(f"Processing synchronously: {file.filename}")
                results = xasr_engine.process_file(tmp_path)
                segments = []
                for i, r in enumerate(results):
                    seg = _result_to_dict(r, i + 1)
                    # Also encode audio for synchronous path
                    audio_b64 = None
                    if r.audio_data is not None and len(r.audio_data) > 0:
                        try:
                            audio_b64 = _numpy_to_wav_base64(r.audio_data, 16000)
                        except Exception as enc_err:
                            logger.warning(f"Audio encoding failed for segment {i}: {enc_err}")
                    seg["audio_wav_base64"] = audio_b64
                    segments.append(seg)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                if xasr_engine.logic_validator:
                    xasr_engine.logic_validator.reset()

                speaker_stats = {}
                for r in results:
                    sid = getattr(r, 'speaker_id', 'unknown')
                    speaker_stats[sid] = speaker_stats.get(sid, 0) + 1
                meeting_domain = xasr_engine.get_meeting_domain() if getattr(xasr_engine, 'enable_cognitive', False) else None
                meeting_hotwords = xasr_engine.get_meeting_hotwords() if getattr(xasr_engine, 'enable_cognitive', False) else []

                sync_data = {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "completed",
                    "engine": "X-ASR (sherpa-onnx zipformer2 v4.5)",
                    "segments_count": len(segments),
                    "segments": segments,
                    "summary": None,
                    "domain": meeting_domain,
                    "hotwords": meeting_hotwords[:20] if meeting_hotwords else [],
                    "speaker_stats": speaker_stats,
                }
                _update_upload_task(
                    file_id,
                    status="completed",
                    progress_stage="done",
                    progress_fraction=1.0,
                    segments_count=len(segments),
                    segments=segments,
                    summary=None,
                    domain=meeting_domain,
                    hotwords=sync_data["hotwords"],
                    speaker_stats=speaker_stats,
                )
                logger.info(f"Done: {len(segments)} segments from {file.filename}")
                return sync_data
        else:
            # ── X-ASR unavailable ──────────────────────────────
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            logger.warning(f"X-ASR not available for {file.filename}, returning demo")
            demo_data = {
                "file_id": file_id,
                "filename": file.filename,
                "status": "demo_mode",
                "engine": "Demo (model not loaded)",
                "message": "Place ONNX models in backend/xasr/models/",
                "demo_data": DEMO_MEETING,
            }
            _update_upload_task(
                file_id,
                status="demo_mode",
                progress_stage="demo_mode",
                progress_fraction=1.0,
                segments_count=len(DEMO_MEETING.get("segments", [])),
                segments=DEMO_MEETING.get("segments", []),
                summary=DEMO_MEETING.get("summary"),
                hotwords=DEMO_MEETING.get("hotwords", []),
                speaker_stats={},
            )
            if queue:
                await _broadcast_upload(file_id, {"type": "complete", "data": demo_data})

            return demo_data

    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        error_file_id = file_id or str(uuid.uuid4())
        _update_upload_task(error_file_id, status="error", error=str(e), progress_stage="error")
        return {
            "file_id": error_file_id,
            "filename": file.filename if file else "unknown",
            "status": "error",
            "error": str(e),
        }


# ===========================================================================
# WebSocket endpoints
# ===========================================================================

@app.websocket("/ws/meeting")
async def ws_meeting(websocket: WebSocket):
    """Demo meeting playback via WebSocket."""
    await manager.connect(websocket)
    logger.info("Demo WebSocket connected")
    try:
        await manager.send_json(websocket, {
            "type": "meeting_start",
            "data": {
                "title": DEMO_MEETING["title"],
                "date": DEMO_MEETING["date"],
                "participants": DEMO_MEETING["participants"],
                "hotwords": DEMO_MEETING["hotwords"],
            }
        })

        for i, seg in enumerate(DEMO_MEETING["segments"]):
            await asyncio.sleep(0.3)
            await manager.send_json(websocket, {
                "type": "transcript_segment",
                "data": {
                    "segment": seg,
                    "segment_index": i,
                    "total_segments": len(DEMO_MEETING["segments"]),
                    "current_time": f"{int(seg['end']//60):02d}:{int(seg['end']%60):02d}",
                    "cumulative_stats": {
                        "segments_processed": i + 1,
                        "logic_flags": sum(1 for s in DEMO_MEETING["segments"][:i+1] if s.get("logic_flags")),
                        "low_confidence": sum(1 for s in DEMO_MEETING["segments"][:i+1] if s.get("uncertain_spans")),
                        "corrections": sum(len(s.get("corrections", [])) for s in DEMO_MEETING["segments"][:i+1]),
                    }
                }
            })

        await asyncio.sleep(0.5)
        await manager.send_json(websocket, {
            "type": "meeting_summary", "data": DEMO_MEETING["summary"]
        })
        await manager.send_json(websocket, {
            "type": "meeting_end",
            "data": {"total_duration": DEMO_MEETING["duration"], "stats": DEMO_MEETING["summary"]["stats"]}
        })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.debug("Demo WebSocket disconnected")


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Real-time audio streaming with X-ASR."""
    await manager.connect(websocket)
    logger.info("Live WebSocket connected")

    engine = None
    if xasr_engine and xasr_engine.is_model_available:
        engine = XASREngine(
            hotwords=DEMO_MEETING["hotwords"],
            enable_logic_validation=True,
            enable_hotword_correction=True,
            enable_uncertainty=True,
            enable_endpoint_detection=False,
            provider="cpu",
            num_threads=1,
        )

    try:
        backend_type = "X-ASR v4.5" if (engine and engine.is_model_available) else "Demo"
        await manager.send_json(websocket, {
            "type": "ready",
            "engine": backend_type,
            "message": f"DiTing ready ({backend_type})",
        })

        if engine and engine.is_model_available:
            engine.start_session()
            logger.info("Live session started, waiting for audio chunks...")
            chunk_count = 0
            send_count = 0
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("action") == "process_chunk":
                    audio_b64 = msg.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                        chunk_count += 1

                        result = await asyncio.to_thread(engine.process_chunk, audio_chunk)

                        if result.text or result.is_final:
                            display_text = str(result.text)
                            postproc_info = None
                            if result.is_final and result.text and HAS_POSTPROCESSOR:
                                try:
                                    display_text, postproc_info = await asyncio.to_thread(
                                        postprocess_text, str(result.text)
                                    )
                                except Exception as pe:
                                    logger.warning(f"Post-processing failed: {pe}")

                            if result.text:
                                logger.info(f"Live chunk #{chunk_count}: text='{result.text[:50]}', final={result.is_final}")
                            resp_data = {
                                "timestamp": float(result.timestamp),
                                "text": str(display_text),
                                "raw_text": str(result.raw_text),
                                "is_partial": bool(result.is_partial),
                                "is_final": bool(result.is_final),
                                "snr_db": float(result.snr_db) if result.snr_db else 25.0,
                                "quality_label": str(result.quality_label),
                                "asr_confidence": float(result.asr_confidence) if result.asr_confidence else 0.8,
                                "speaker_id": getattr(result, 'speaker_id', 'unknown'),
                                "corrections": result.corrections or [],
                                "logic_flags": result.logic_flags or [],
                                "terms": result.terms or [],
                                "uncertain_spans": result.uncertain_spans or [],
                            }
                            if postproc_info:
                                resp_data["postprocessed"] = True
                                resp_data["original_text"] = postproc_info.get('original_text', '')
                                resp_data["fillers_removed"] = postproc_info.get('fillers_removed', [])
                                resp_data["corrections"] = postproc_info.get('corrections', [])
                            await manager.send_json(websocket, {
                                "type": "live_result",
                                "data": resp_data
                            })
                            send_count += 1
                elif msg.get("action") == "stop":
                    logger.info(f"Live session stopped: chunks={chunk_count}, sent={send_count}")
                    engine.end_session()
                    break
        else:
            # Demo fallback
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("action") == "stop":
                    break
                await manager.send_json(websocket, {
                    "type": "live_result",
                    "data": {
                        "timestamp": time.time(),
                        "text": "[Demo] X-ASR model not loaded",
                        "is_partial": True,
                        "is_final": False,
                        "snr_db": 22,
                        "quality_label": "medium",
                        "asr_confidence": 0.75,
                    }
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Live WS error: {e}")
        try:
            await manager.send_json(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if engine:
            engine.end_session()


@app.websocket("/ws/upload/{file_id}")
async def ws_upload_progress(websocket: WebSocket, file_id: str):
    """Real-time upload progress via WebSocket."""
    await websocket.accept()
    queue = asyncio.Queue()
    upload_sessions.setdefault(file_id, set()).add(queue)
    logger.info(f"Upload WS connected: {file_id[:8]}...")

    try:
        task = _get_upload_task(file_id)
        if task:
            await websocket.send_json({
                "type": "snapshot",
                "data": {"task": _serialize_upload_task(task, include_segments=True)}
            })
            if task.get("status") == "completed":
                await websocket.send_json({"type": "complete", "data": _serialize_upload_task(task, include_segments=True)})
                return
            if task.get("status") == "error":
                await websocket.send_json({"type": "error", "data": {"message": task.get("error") or "Upload failed"}})
                return
            if task.get("status") == "cancelled":
                await websocket.send_json({"type": "cancelled", "data": {"file_id": file_id, "status": "cancelled"}})
                return

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=600)
                await websocket.send_json(msg)
                if msg["type"] in ("complete", "error", "cancelled"):
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "timeout",
                    "data": {"message": "Processing timeout (10 min)"}
                })
                break

    except WebSocketDisconnect:
        logger.debug(f"Upload WS disconnected: {file_id[:8]}...")
    except Exception as e:
        logger.error(f"Upload WS error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass
    finally:
        queues = upload_sessions.get(file_id)
        if queues:
            queues.discard(queue)
            if not queues:
                upload_sessions.pop(file_id, None)


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """
    Stream recent logs to frontend for debugging.
    Sends initial batch, then periodic updates.
    """
    await websocket.accept()
    logger.info("Log streaming WS connected")
    try:
        while True:
            logs = get_recent_log_entries(50)
            await websocket.send_json({"type": "logs", "data": logs})
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=2)
                msg = json.loads(data)
                if msg.get("action") == "stop":
                    break
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Log WS error: {e}")


# ===========================================================================
# Static files
# ===========================================================================

FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "frontend", "dist")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/app")
async def serve_spa():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not built.</h1>")


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  DiTing v4.5 - Smart Meeting Speech Cognitive System")
    print("  Backend: http://localhost:8765")
    print("  API Docs: http://localhost:8765/docs")
    print("  Logs: backend/logs/diting.log")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
