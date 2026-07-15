"""
谛听 (DiTing) - Smart Meeting Speech Cognitive System
Backend Server v2.0 (2026-07-15)
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ───────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# ── Logging ─────────────────────────────────────────────────────
from utils.logger import init_logging, get_logger, get_recent_logs, log_buffer
from modules import supabase_db as db

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
    logger.info("Connected to Supabase: %s", os.environ.get("SUPABASE_URL", "not configured"))
    try:
        db._get_client().table("friends").select("*").limit(1).execute()
        logger.info("Supabase friends table: OK")
    except Exception as e:
        logger.error("Supabase friends table MISSING: %s", e)
    threading.Thread(target=_load_xasr_engine, daemon=True).start()
    yield
    logger.info("Shutting down DiTing backend...")

app = FastAPI(
    title="DiTing - Smart Meeting Speech Cognitive System",
    version="2.0.0",
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

# ===========================================================================
# API Routes
# ===========================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "DiTing v2.0",
        "xasr_available": HAS_XASR and xasr_engine is not None and xasr_engine.is_model_available,
        "xasr_loading": xasr_loading,
        "eval_available": HAS_EVAL,
        "timestamp": time.time(),
    }


def _token_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _require_user(authorization: str | None):
    user = db.get_user_by_token(_token_from_header(authorization))
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


# ===========================================================================
# User management / auth / meeting reservation endpoints
# ===========================================================================

@app.post("/api/auth/login")
async def auth_login(data: dict):
    result = db.login(data.get("username", ""), data.get("password", ""))
    if not result:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return result


@app.post("/api/auth/register")
async def auth_register(data: dict):
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")
    try:
        user = db.create_user({
            "username": username,
            "display_name": data.get("display_name") or username,
            "password": password,
            "role": "user",
        })
        return {"user": user, "message": "注册成功"}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/auth/me")
async def auth_me(authorization: str | None = Header(None)):
    return {"user": _require_user(authorization)}


@app.post("/api/auth/logout")
async def auth_logout(authorization: str | None = Header(None)):
    db.logout(_token_from_header(authorization))
    return {"ok": True}


@app.get("/api/users")
async def api_list_users(authorization: str | None = Header(None)):
    _require_user(authorization)
    return {"users": db.list_users()}


@app.post("/api/users")
async def api_create_user(data: dict, authorization: str | None = Header(None)):
    user = _require_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    try:
        return {"user": db.create_user(data)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/api/users/{user_id}")
async def api_update_user(user_id: str, data: dict, authorization: str | None = Header(None)):
    user = _require_user(authorization)
    if user.get("role") != "admin" and user.get("id") != user_id:
        raise HTTPException(status_code=403, detail="Permission denied")
    return {"user": db.update_user(user_id, data)}


@app.get("/api/meetings/reservations")
async def api_list_reservations(
    user_id: str | None = Query(None),
    status: str | None = Query(None),
    authorization: str | None = Header(None),
):
    _require_user(authorization)
    reservations = db.list_reservations(user_id=user_id, status=status)
    now = __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc)
    from datetime import datetime as _dt
    for r in reservations:
        try:
            st = _dt.fromisoformat(r["start_time"].replace("Z", "+00:00"))
            et = _dt.fromisoformat(r["end_time"].replace("Z", "+00:00"))
            if now < st:
                r["time_status"] = "未开始"
            elif now > et:
                r["time_status"] = "已结束"
            else:
                r["time_status"] = "进行中"
        except Exception:
            r["time_status"] = "未知"
    return {"reservations": reservations}


@app.post("/api/meetings/reservations")
async def api_create_reservation(data: dict, authorization: str | None = Header(None)):
    user = _require_user(authorization)
    data.setdefault("organizer_user_id", user["id"])
    try:
        return {"reservation": db.create_reservation(data)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/meetings/reservations/{reservation_id}")
async def api_update_reservation(reservation_id: str, data: dict, authorization: str | None = Header(None)):
    _require_user(authorization)
    return {"reservation": db.update_reservation(reservation_id, data)}


@app.post("/api/meetings/join")
async def api_join_meeting(data: dict, authorization: str | None = Header(None)):
    user = db.get_user_by_token(_token_from_header(authorization))
    if user:
        data.setdefault("user_id", user["id"])
        data.setdefault("display_name", user["display_name"])
    try:
        return db.join_meeting(data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ===========================================================================
# Friends endpoints
# ===========================================================================

@app.get("/api/friends/search")
async def api_search_users(q: str = Query(...), authorization: str | None = Header(None)):
    user = db.get_user_by_token(_token_from_header(authorization))
    exclude = user["id"] if user else None
    return {"users": db.search_users(q, exclude_user_id=exclude)}


@app.get("/api/friends")
async def api_list_friends(authorization: str | None = Header(None)):
    user = _require_user(authorization)
    return {"friends": db.list_friends(user["id"])}


@app.post("/api/friends")
async def api_add_friend(data: dict, authorization: str | None = Header(None)):
    user = _require_user(authorization)
    friend_id = data.get("friend_id")
    if not friend_id:
        raise HTTPException(status_code=400, detail="friend_id required")
    result = db.add_friend(user["id"], friend_id)
    if not result:
        raise HTTPException(status_code=409, detail="已经是好友或添加失败")
    return {"friend": result}


@app.delete("/api/friends/{friend_id}")
async def api_remove_friend(friend_id: str, authorization: str | None = Header(None)):
    user = _require_user(authorization)
    db.remove_friend(user["id"], friend_id)
    return {"ok": True}


# ===========================================================================
# Meeting Analysis endpoints
# ===========================================================================

@app.post("/api/meetings/analysis")
async def api_save_analysis(data: dict, authorization: str | None = Header(None)):
    user = db.get_user_by_token(_token_from_header(authorization))
    if user:
        data.setdefault("created_by", user["id"])
    try:
        analysis = db.save_analysis(data)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meetings/analysis")
async def api_list_analyses(
    meeting_id: str | None = Query(None),
    authorization: str | None = Header(None),
):
    user = db.get_user_by_token(_token_from_header(authorization))
    user_id = user["id"] if user else None
    return {"analyses": db.list_analyses(user_id=user_id, meeting_id=meeting_id)}


@app.get("/api/meetings/analysis/{analysis_id}")
async def api_get_analysis(analysis_id: str, authorization: str | None = Header(None)):
    _require_user(authorization)
    analysis = db.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"analysis": analysis}


@app.delete("/api/meetings/analysis/{analysis_id}")
async def api_delete_analysis(analysis_id: str, authorization: str | None = Header(None)):
    _require_user(authorization)
    db.delete_analysis(analysis_id)
    return {"ok": True}


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
async def get_recent_logs(n: int = Query(50, ge=1, le=500)):
    """Get recent log entries (for frontend debug panel)."""
    return {"logs": get_recent_logs(n), "count": len(log_buffer)}


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

        if xasr_engine and xasr_engine.is_model_available:
            if queue:
                # ── WebSocket streaming mode ──────────────────
                await queue.put({
                    "type": "status",
                    "data": {
                        "status": "processing",
                        "message": f"Processing {file.filename} ({file_size_mb:.1f}MB)...",
                        "filename": file.filename,
                        "file_id": file_id,
                        "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
                    }
                })

                loop = asyncio.get_event_loop()

                def on_segment(result, idx, total):
                    seg_data = _result_to_dict(result, idx)
                    # Encode raw audio to base64 WAV for frontend waveform
                    audio_b64 = None
                    if result.audio_data is not None and len(result.audio_data) > 0:
                        try:
                            audio_b64 = _numpy_to_wav_base64(result.audio_data, 16000)
                        except Exception as enc_err:
                            logger.warning(f"Audio encoding failed for segment {idx}: {enc_err}")
                    seg_data["audio_wav_base64"] = audio_b64

                    asyncio.run_coroutine_threadsafe(
                        queue.put({
                            "type": "segment",
                            "data": {
                                "segment": seg_data,
                                "segment_index": idx - 1,
                                "total_estimated": total,
                                "cumulative_stats": {"segments_processed": idx},
                            }
                        }),
                        loop
                    )

                def on_progress(stage, fraction):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({
                            "type": "progress",
                            "data": {"stage": stage, "fraction": fraction}
                        }),
                        loop
                    )

                def do_process():
                    try:
                        results = xasr_engine.process_file(
                            tmp_path, on_segment=on_segment, on_progress=on_progress
                        )
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
                        asyncio.run_coroutine_threadsafe(
                            queue.put({
                                "type": "complete",
                                "data": {
                                    "file_id": file_id,
                                    "filename": file.filename,
                                    "status": "completed",
                                    "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
                                    "segments_count": len(results),
                                    "segments": final_segments,
                                }
                            }),
                            loop
                        )
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        logger.error(traceback.format_exc())
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "error", "data": {"message": str(e)}}),
                            loop
                        )
                    finally:
                        if xasr_engine.logic_validator:
                            xasr_engine.logic_validator.reset()
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                executor.submit(do_process)

                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "processing",
                    "size_mb": round(file_size_mb, 2),
                    "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
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

                logger.info(f"Done: {len(segments)} segments from {file.filename}")
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "completed",
                    "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
                    "segments_count": len(segments),
                    "segments": segments,
                }
        else:
            # ── X-ASR unavailable ──────────────────────────────
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            logger.warning(f"X-ASR not available for {file.filename}, returning demo")
            if queue:
                await queue.put({
                    "type": "complete",
                    "data": {
                        "file_id": file_id,
                        "filename": file.filename,
                        "status": "demo_mode",
                        "engine": "Demo (model not loaded)",
                        "message": "Place ONNX models in backend/xasr/models/",
                        "demo_data": DEMO_MEETING,
                    }
                })

            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "demo_mode",
                "engine": "Demo (model not loaded)",
                "demo_data": DEMO_MEETING,
            }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return {
            "file_id": file_id or str(uuid.uuid4()),
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
            enable_endpoint_detection=True,
            provider="cpu",
            num_threads=1,
        )

    try:
        backend_type = "X-ASR v2.0" if (engine and engine.is_model_available) else "Demo"
        await manager.send_json(websocket, {
            "type": "ready",
            "engine": backend_type,
            "message": f"DiTing ready ({backend_type})",
        })

        if engine and engine.is_model_available:
            engine.start_session()
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("action") == "process_chunk":
                    audio_b64 = msg.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                        result = engine.process_chunk(audio_chunk)

                        await manager.send_json(websocket, {
                            "type": "live_result",
                            "data": {
                                "timestamp": result.timestamp,
                                "text": result.text,
                                "raw_text": result.raw_text,
                                "is_partial": result.is_partial,
                                "is_final": result.is_final,
                                "snr_db": result.snr_db,
                                "quality_label": result.quality_label,
                                "asr_confidence": result.asr_confidence,
                                "corrections": result.corrections,
                                "logic_flags": result.logic_flags,
                                "terms": result.terms,
                                "uncertain_spans": result.uncertain_spans,
                            }
                        })
                elif msg.get("action") == "finalize":
                    result = engine._finalize_results()
                    if result:
                        await manager.send_json(websocket, {
                            "type": "live_result",
                            "data": {
                                "timestamp": result.timestamp,
                                "text": result.text,
                                "is_partial": False, "is_final": True,
                                "snr_db": result.snr_db,
                                "quality_label": result.quality_label,
                                "asr_confidence": result.asr_confidence,
                                "logic_flags": result.logic_flags,
                                "terms": result.terms,
                            }
                        })
                elif msg.get("action") == "stop":
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
                        "snr_estimate": 22,
                        "partial_text": "[Demo] X-ASR model not loaded",
                        "confidence": 0.75,
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
    upload_sessions[file_id] = queue
    logger.info(f"Upload WS connected: {file_id[:8]}...")

    try:
        await websocket.send_json({
            "type": "connected",
            "data": {
                "file_id": file_id,
                "message": "WebSocket connected, waiting for upload...",
            }
        })

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=600)
                await websocket.send_json(msg)
                if msg["type"] in ("complete", "error"):
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
            logs = get_recent_logs(50)
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
    print("  DiTing v2.0 - Smart Meeting Speech Cognitive System")
    print("  Backend: http://localhost:8765")
    print("  API Docs: http://localhost:8765/docs")
    print("  Logs: backend/logs/diting.log")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
