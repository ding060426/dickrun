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
import threading
import concurrent.futures
import traceback
from functools import partial
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ───────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# ── Logging ─────────────────────────────────────────────────────
from utils.logger import init_logging, get_logger, get_recent_logs, log_buffer
from upload_storage import UploadTooLargeError, save_upload_to_temp
from xasr.hotword_config import HotwordConfigStore
from xasr.runtime_config import RuntimeConfigStore
from diarization import (
    ChunkedDiarizationBackend,
    ChunkedDiarizationConfig,
    OfflineMeetingPipeline,
    SherpaDiarizationBackend,
    SherpaSpeakerEmbedder,
)
from diarization.registry import MeetingRegistry
from build_info import API_REVISION
from modules.management_store import select_management_store

init_logging(console_level="INFO", file_level="DEBUG")
logger = get_logger("main")
db = select_management_store()

# ── X-ASR ───────────────────────────────────────────────────────
try:
    from xasr.asr_engine import XASREngine, ASRResult
    from xasr.engine_pool import AsrEnginePool
    from xasr.file_vad import SileroFileVad
    from xasr.live_audio import (
        LiveAudioProfile,
        LiveAudioProtocolError,
        LiveAudioSession,
        create_live_vad,
        find_silero_vad_model,
        get_live_endpoint_grace_ms,
        get_live_audio_profile,
    )
    from xasr.recording import LiveRecording
    HAS_XASR = True
    logger.info("X-ASR module loaded")
except ImportError as e:
    logger.warning(f"X-ASR module load failed: {e}")
    logger.warning("Transcription endpoints will report X-ASR as unavailable")
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
final_xasr_engine: Optional[XASREngine] = None
xasr_pool: Optional[object] = None
xasr_loading: bool = False
_xasr_reload_lock = threading.Lock()
_xasr_reload_pending = False
_xasr_reload_worker_active = False
PROCESSING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("DITING_PROCESSING_WORKERS", "2"))),
    thread_name_prefix="diting-asr",
)
RECORDINGS_DIR = Path(
    os.getenv("DITING_RECORDINGS_DIR", os.path.join(BACKEND_DIR, "recordings"))
)
HOTWORDS_CONFIG_PATH = Path(
    os.getenv("DITING_HOTWORDS_CONFIG", os.path.join(BACKEND_DIR, "data", "hotwords.json"))
)
RUNTIME_CONFIG_PATH = Path(
    os.getenv("DITING_RUNTIME_CONFIG", os.path.join(BACKEND_DIR, "data", "settings.json"))
)
hotword_config_store = None
runtime_config_store = RuntimeConfigStore(RUNTIME_CONFIG_PATH)
DIARIZATION_MODEL_DIR = Path(BACKEND_DIR) / "diarization" / "models"
base_diarization_backend = SherpaDiarizationBackend(
    os.getenv(
        "DITING_DIARIZATION_SEGMENTATION_MODEL",
        str(DIARIZATION_MODEL_DIR / "pyannote-segmentation-3.0.int8.onnx"),
    ),
    os.getenv(
        "DITING_SPEAKER_EMBEDDING_MODEL",
        str(DIARIZATION_MODEL_DIR / "3dspeaker-eres2net.onnx"),
    ),
    threshold=float(os.getenv("DITING_DIARIZATION_THRESHOLD", "0.8")),
    num_threads=int(os.getenv("DITING_DIARIZATION_THREADS", "4")),
    min_turn_sec=float(os.getenv("DITING_DIARIZATION_MIN_TURN_SEC", "0.5")),
)
diarization_chunk_config = ChunkedDiarizationConfig(
    enabled=os.getenv("DITING_DIARIZATION_CHUNKING", "true").lower()
    not in {"0", "false", "no", "off"},
    long_audio_threshold_sec=float(
        os.getenv("DITING_DIARIZATION_LONG_AUDIO_SEC", "600")
    ),
    target_chunk_sec=float(
        os.getenv("DITING_DIARIZATION_TARGET_CHUNK_SEC", "300")
    ),
    max_chunk_sec=float(os.getenv("DITING_DIARIZATION_MAX_CHUNK_SEC", "480")),
    overlap_sec=float(os.getenv("DITING_DIARIZATION_CHUNK_OVERLAP_SEC", "2")),
    silence_search_sec=float(
        os.getenv("DITING_DIARIZATION_SILENCE_SEARCH_SEC", "15")
    ),
    skip_silence_sec=float(
        os.getenv("DITING_DIARIZATION_SKIP_SILENCE_SEC", "20")
    ),
    max_workers=int(os.getenv("DITING_DIARIZATION_MAX_WORKERS", "2")),
    worker_threads=int(os.getenv("DITING_DIARIZATION_WORKER_THREADS", "2")),
    stitch_threshold=float(os.getenv("DITING_SPEAKER_STITCH_THRESHOLD", "0.75")),
)
diarization_speech_detector = None
diarization_vad_model = Path(BACKEND_DIR) / "xasr" / "models" / "silero_vad.onnx"
if HAS_XASR and diarization_vad_model.is_file():
    diarization_speech_detector = SileroFileVad(
        diarization_vad_model,
        num_threads=1,
    )
diarization_speaker_embedder = SherpaSpeakerEmbedder(
    base_diarization_backend.embedding_model,
    num_threads=1,
)
diarization_backend = ChunkedDiarizationBackend(
    base_diarization_backend,
    worker_factory=lambda: base_diarization_backend.spawn(
        num_threads=diarization_chunk_config.worker_threads
    ),
    speech_detector=diarization_speech_detector,
    speaker_embedder=diarization_speaker_embedder,
    config=diarization_chunk_config,
)
meeting_pipeline = OfflineMeetingPipeline(diarization_backend)
meeting_registry = MeetingRegistry()

def _load_xasr_engine():
    """Build and atomically publish the configured live/final ASR runtimes."""
    global xasr_engine, final_xasr_engine, xasr_pool, xasr_loading
    if not HAS_XASR:
        logger.info("X-ASR not available; transcription is disabled")
        return
    xasr_loading = True
    logger.info("Loading configured X-ASR live/final runtimes...")
    try:
        runtime_settings = runtime_config_store.load()
        hotword_settings = hotword_config_store.load()
        pool = AsrEnginePool(
            XASREngine.DEFAULT_MODEL_DIR,
            base_options={
                "enable_logic_validation": True,
                "enable_uncertainty": True,
                "enable_endpoint_detection": False,
                "provider": "cpu",
                "num_threads": 2,
            },
        )
        status = pool.reload(runtime_settings["recognition"], hotword_settings)
        xasr_pool = pool
        xasr_engine = pool.live_engine
        final_xasr_engine = pool.final_engine
        logger.info(
            "X-ASR ready: live=%sms final=%sms shared=%s file_vad=%s",
            status["live"]["chunk_ms"],
            status["final"]["chunk_ms"],
            status["shared_runtime"],
            status["file_vad_provider"],
        )
    except Exception as e:
        logger.error(f"X-ASR init failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        xasr_loading = False


def _xasr_reload_worker() -> None:
    """Serialize heavyweight reloads and coalesce saves to the latest config."""
    global _xasr_reload_pending, _xasr_reload_worker_active
    while True:
        with _xasr_reload_lock:
            if not _xasr_reload_pending:
                _xasr_reload_worker_active = False
                return
            _xasr_reload_pending = False
        _load_xasr_engine()


def _schedule_xasr_reload() -> None:
    global _xasr_reload_pending, _xasr_reload_worker_active
    with _xasr_reload_lock:
        _xasr_reload_pending = True
        if _xasr_reload_worker_active:
            return
        _xasr_reload_worker_active = True
    threading.Thread(target=_xasr_reload_worker, daemon=True).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: fire-and-forget model loading."""
    logger.info("Starting DiTing backend server...")
    try:
        db.init_db()
        logger.info("Meeting management store ready: %s", db.__name__)
    except Exception as error:
        logger.error("Meeting management store initialization failed: %s", error)
    _schedule_xasr_reload()
    yield
    logger.info("Shutting down DiTing backend...")
    PROCESSING_EXECUTOR.shutdown(wait=False, cancel_futures=True)

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

hotword_config_store = HotwordConfigStore(
    HOTWORDS_CONFIG_PATH,
    [],
)

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
    vad_model = find_silero_vad_model() if HAS_XASR else None
    diarization_status = meeting_pipeline.status()
    return {
        "status": "ok",
        "service": "DiTing v2.0",
        "api_revision": API_REVISION,
        "xasr_available": HAS_XASR and xasr_engine is not None and xasr_engine.is_model_available,
        "xasr_loading": xasr_loading,
        "live_vad_available": vad_model is not None,
        "diarization_available": diarization_status["available"],
        "management_store": db.__name__.rsplit(".", 1)[-1],
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
# Meeting management routes
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
        user = db.create_user(
            {
                "username": username,
                "display_name": data.get("display_name") or username,
                "password": password,
                "role": "user",
            }
        )
        return {"user": user, "message": "注册成功"}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.put("/api/users/{user_id}")
async def api_update_user(
    user_id: str,
    data: dict,
    authorization: str | None = Header(None),
):
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
    now = datetime.now().astimezone()
    for reservation in reservations:
        try:
            start = datetime.fromisoformat(
                reservation["start_time"].replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                reservation["end_time"].replace("Z", "+00:00")
            )
            if start.tzinfo is None:
                start = start.replace(tzinfo=now.tzinfo)
            if end.tzinfo is None:
                end = end.replace(tzinfo=now.tzinfo)
            if now < start:
                reservation["time_status"] = "未开始"
            elif now > end:
                reservation["time_status"] = "已结束"
            else:
                reservation["time_status"] = "进行中"
        except (AttributeError, KeyError, TypeError, ValueError):
            reservation["time_status"] = "未知"
    return {"reservations": reservations}


@app.post("/api/meetings/reservations")
async def api_create_reservation(
    data: dict,
    authorization: str | None = Header(None),
):
    user = _require_user(authorization)
    data.setdefault("organizer_user_id", user["id"])
    try:
        return {"reservation": db.create_reservation(data)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.put("/api/meetings/reservations/{reservation_id}")
async def api_update_reservation(
    reservation_id: str,
    data: dict,
    authorization: str | None = Header(None),
):
    _require_user(authorization)
    return {"reservation": db.update_reservation(reservation_id, data)}


@app.post("/api/meetings/join")
async def api_join_meeting(
    data: dict,
    authorization: str | None = Header(None),
):
    user = db.get_user_by_token(_token_from_header(authorization))
    if user:
        data.setdefault("user_id", user["id"])
        data.setdefault("display_name", user["display_name"])
    try:
        return db.join_meeting(data)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/friends/search")
async def api_search_users(
    q: str = Query(...),
    authorization: str | None = Header(None),
):
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
async def api_remove_friend(
    friend_id: str,
    authorization: str | None = Header(None),
):
    user = _require_user(authorization)
    db.remove_friend(user["id"], friend_id)
    return {"ok": True}


@app.post("/api/meetings/analysis")
async def api_save_analysis(
    data: dict,
    authorization: str | None = Header(None),
):
    user = _require_user(authorization)
    data.setdefault("created_by", user["id"])
    try:
        return {"analysis": db.save_analysis(data)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/meetings/analysis")
async def api_list_analyses(
    meeting_id: str | None = Query(None),
    authorization: str | None = Header(None),
):
    user = _require_user(authorization)
    return {
        "analyses": db.list_analyses(
            user_id=user["id"],
            meeting_id=meeting_id,
        )
    }


@app.get("/api/meetings/analysis/{analysis_id}")
async def api_get_analysis(
    analysis_id: str,
    authorization: str | None = Header(None),
):
    user = _require_user(authorization)
    analysis = db.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if (
        analysis.get("created_by")
        and analysis["created_by"] != user["id"]
        and user.get("role") != "admin"
    ):
        raise HTTPException(status_code=403, detail="Permission denied")
    return {"analysis": analysis}


@app.delete("/api/meetings/analysis/{analysis_id}")
async def api_delete_analysis(
    analysis_id: str,
    authorization: str | None = Header(None),
):
    user = _require_user(authorization)
    analysis = db.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if (
        analysis.get("created_by")
        and analysis["created_by"] != user["id"]
        and user.get("role") != "admin"
    ):
        raise HTTPException(status_code=403, detail="Permission denied")
    db.delete_analysis(analysis_id)
    return {"ok": True}


@app.get("/api/xasr/status")
async def get_xasr_status():
    """Get X-ASR engine status with detail."""
    if not HAS_XASR:
        return {"available": False, "reason": "X-ASR module not installed", "loading": False}
    if xasr_engine is None:
        return {
            "available": HAS_XASR,
            "reason": "Loading model..." if xasr_loading else "Engine init failed",
            "model_available": False,
            "loading": xasr_loading,
        }

    vad_model = find_silero_vad_model(xasr_engine.model_dir)
    runtime_settings = runtime_config_store.load()
    pool_status = xasr_pool.status() if xasr_pool else {}
    return {
        "available": True,
        "model_available": xasr_engine.is_model_available,
        "model_dir": xasr_engine.model_dir,
        "endpoint_detection": xasr_engine.enable_endpoint_detection,
        "live_vad": {
            "provider": "sherpa-silero-vad" if vad_model else "asr-endpoint-fallback",
            "available": vad_model is not None,
            "model_path": str(vad_model) if vad_model else None,
            "endpoint_policy": (
                "silero-vad-with-resume-grace"
                if vad_model else "sherpa-asr-endpoint-fallback"
            ),
            "endpoint_grace_ms": runtime_settings["microphone"]["endpoint_grace_ms"] if vad_model else 0,
        },
        "models": pool_status,
        "file_vad": {
            "provider": pool_status.get("file_vad_provider", "unavailable"),
            "threshold": runtime_settings["recognition"]["file_vad_threshold"],
        },
        "diarization": meeting_pipeline.status(),
        "hotwords_count": hotword_config_store.load()["active_count"],
        "features": {
            "logic_validation": xasr_engine.enable_logic_validation,
            "hotword_correction": xasr_engine.enable_hotword_correction,
            "fuzzy_pinyin": xasr_engine.enable_fuzzy_pinyin,
            "uncertainty_estimation": xasr_engine.enable_uncertainty,
            "speaker_diarization": meeting_pipeline.status()["available"],
        },
        "loading": xasr_loading,
    }


@app.get("/api/hotwords")
async def get_hotwords():
    """Get persistent hotword settings used by future ASR sessions."""
    return {**hotword_config_store.load(), "applies_to": "new_sessions"}


def _apply_hotword_settings(settings: dict) -> None:
    if not xasr_pool:
        return
    xasr_pool.configure_hotwords(settings)


def _complete_settings_payload() -> dict:
    return {
        **runtime_config_store.load(),
        "hotwords": hotword_config_store.load(),
        "models": xasr_pool.status() if xasr_pool else {
            "available_profiles": [],
            "live": {},
            "final": {},
            "shared_runtime": False,
            "file_vad_provider": "unavailable",
        },
        "applies_to": "new_sessions",
    }


@app.get("/api/settings")
async def get_settings():
    """Return the unified recognition, microphone, and hotword settings."""
    return _complete_settings_payload()


@app.put("/api/settings")
async def replace_settings(data: dict):
    """Persist unified settings and reload model runtimes only when required."""
    previous = runtime_config_store.load()
    runtime_payload = {
        "recognition": data.get("recognition", previous["recognition"]),
        "microphone": data.get("microphone", previous["microphone"]),
    }
    saved_runtime = runtime_config_store.save(runtime_payload)
    if "hotwords" in data:
        saved_hotwords = hotword_config_store.save(data["hotwords"])
        _apply_hotword_settings(saved_hotwords)

    reload_required = previous["recognition"] != saved_runtime["recognition"]
    if reload_required:
        _schedule_xasr_reload()
    return {
        **_complete_settings_payload(),
        "ok": True,
        "reloading_models": reload_required,
    }


@app.post("/api/hotwords")
async def add_hotwords(data: dict):
    """Compatibility endpoint: add words while preserving current settings."""
    raw_words = data.get("words", [])
    new_words = [
        item.get("text", "") if isinstance(item, dict) else item
        for item in raw_words
    ]
    settings = hotword_config_store.add_words(new_words)
    _apply_hotword_settings(settings)
    return {**settings, "added": new_words, "ok": True, "applies_to": "new_sessions"}


@app.put("/api/hotwords")
async def replace_hotword_settings(data: dict):
    """Persist the complete list, per-word scores, and fuzzy-pinyin switches."""
    settings = hotword_config_store.save(data)
    _apply_hotword_settings(settings)
    return {**settings, "ok": True, "applies_to": "new_sessions"}


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
        "speaker_id": result.speaker_id,
        "speaker_name": result.speaker_name,
        "speaker_confidence": round(float(result.speaker_confidence), 3),
        "overlap": bool(result.overlap),
        "overlap_speakers": result.overlap_speakers,
        "words": [
            {
                "text": word.text,
                "start_sec": word.start_sec,
                "end_sec": word.end_sec,
                "speaker_id": word.speaker_id,
                "confidence": word.confidence,
            }
            for word in result.words
        ],
        "asr_confidence": round(float(result.asr_confidence), 3),
        "snr_db": round(float(result.snr_db), 1),
        "rt60": round(float(result.rt60), 3),
        "quality_score": round(float(result.quality_score), 3),
        "quality_label": result.quality_label,
        "corrections": result.corrections,
        "postprocessed": bool(result.postprocessed),
        "original_text": result.original_text,
        "fillers_removed": result.fillers_removed,
        "repetitions_merged": result.repetitions_merged,
        "logic_flags": result.logic_flags,
        "terms": result.terms,
        "data_points": result.data_points,
        "uncertain_spans": result.uncertain_spans,
        "uncertainty": result.uncertainty,
    })


def _live_result_to_dict(result: ASRResult) -> dict:
    """Convert a streaming result while preserving partial/final semantics."""
    payload = _result_to_dict(result, 0)
    payload.update({
        "timestamp": float(result.timestamp),
        "is_partial": bool(result.is_partial),
        "is_final": bool(result.is_final),
    })
    return payload


@app.post("/api/audio/upload")
async def upload_audio(
    file: UploadFile = File(...),
    file_id: str = Query(None),
    enable_diarization: bool = Query(True),
    num_speakers: int | None = Query(None, ge=1, le=20),
):
    """Upload audio file for X-ASR processing with real-time WebSocket progress."""
    tmp_path = None
    try:
        if not file_id:
            file_id = str(uuid.uuid4())

        # Stream to disk in bounded chunks so long meetings do not require an
        # equally large in-memory bytes object.
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        max_upload_mb = max(1, int(os.getenv("DITING_MAX_UPLOAD_MB", "2048")))
        stored = await save_upload_to_temp(
            file,
            suffix=suffix,
            max_bytes=max_upload_mb * 1024 * 1024,
        )
        tmp_path = stored.path

        file_size_mb = stored.size_bytes / (1024 * 1024)
        logger.info(f"Upload: {file.filename} ({file_size_mb:.1f}MB) file_id={file_id[:8]}...")

        queue = upload_sessions.get(file_id)

        if xasr_pool and xasr_engine and xasr_engine.is_model_available:
            processing_engine = xasr_pool.create_final_session()
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
                        run = meeting_pipeline.process_file(
                            tmp_path,
                            processing_engine,
                            enable_diarization=enable_diarization,
                            num_speakers=num_speakers,
                            on_segment=on_segment,
                            on_progress=on_progress,
                        )
                        results = run.results
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
                        meeting_registry.register(
                            file_id,
                            filename=file.filename,
                            segments=final_segments,
                            speakers=run.speakers,
                        )
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
                                    "speakers": run.speakers,
                                    "diarization": run.metadata(),
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
                        if processing_engine.logic_validator:
                            processing_engine.logic_validator.reset()
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                PROCESSING_EXECUTOR.submit(do_process)

                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "processing",
                    "size_mb": round(file_size_mb, 2),
                    "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
                    "diarization_requested": enable_diarization,
                    "num_speakers": num_speakers,
                }
            else:
                # ── Synchronous mode (no WebSocket) ────────────
                logger.info(f"Processing synchronously: {file.filename}")
                loop = asyncio.get_running_loop()
                run = await loop.run_in_executor(
                    PROCESSING_EXECUTOR,
                    partial(
                        meeting_pipeline.process_file,
                        tmp_path,
                        processing_engine,
                        enable_diarization=enable_diarization,
                        num_speakers=num_speakers,
                    ),
                )
                results = run.results
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
                meeting_registry.register(
                    file_id,
                    filename=file.filename,
                    segments=segments,
                    speakers=run.speakers,
                )
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                if processing_engine.logic_validator:
                    processing_engine.logic_validator.reset()

                logger.info(f"Done: {len(segments)} segments from {file.filename}")
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "completed",
                    "engine": "X-ASR (sherpa-onnx zipformer2 v2.0)",
                    "segments_count": len(segments),
                    "segments": segments,
                    "speakers": run.speakers,
                    "diarization": run.metadata(),
                }
        else:
            # ── X-ASR unavailable ──────────────────────────────
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            error_message = "X-ASR model is not loaded; transcription cannot start"
            logger.warning("%s: %s", error_message, file.filename)
            if queue:
                await queue.put({
                    "type": "complete",
                    "data": {
                        "file_id": file_id,
                        "filename": file.filename,
                        "status": "error",
                        "error": error_message,
                    }
                })

            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "error",
                "error": error_message,
            }

    except UploadTooLargeError as e:
        logger.warning("Upload rejected: %s", e)
        return {
            "file_id": file_id or str(uuid.uuid4()),
            "filename": file.filename if file else "unknown",
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return {
            "file_id": file_id or str(uuid.uuid4()),
            "filename": file.filename if file else "unknown",
            "status": "error",
            "error": str(e),
        }


@app.get("/api/meetings/{meeting_id}")
async def get_processed_meeting(meeting_id: str):
    """Return the lightweight transcript metadata retained for speaker edits."""

    meeting = meeting_registry.get(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    return meeting


@app.patch("/api/meetings/{meeting_id}/speakers/{speaker_id}")
async def rename_meeting_speaker(meeting_id: str, speaker_id: str, data: dict):
    """Rename a diarized speaker and update every retained transcript segment."""

    try:
        meeting = meeting_registry.rename(meeting_id, speaker_id, data.get("name", ""))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting or speaker not found")
    return meeting


# ===========================================================================
# WebSocket endpoints
# ===========================================================================

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Low-latency preview plus durable recording and canonical final ASR."""
    await manager.connect(websocket)
    logger.info("Live WebSocket connected")

    engine = (
        xasr_pool.create_live_session()
        if xasr_pool and xasr_engine and xasr_engine.is_model_available
        else None
    )
    session = None
    vad_provider = "unavailable"
    live_profile = None
    stream_id = None
    normal_stop = False
    recovery_path = None

    def build_session(profile_name=None, requested_stream_id=None):
        nonlocal vad_provider, live_profile, stream_id
        microphone_settings = runtime_config_store.load()["microphone"]
        selected_profile = profile_name or microphone_settings["live_profile"]
        base_profile = get_live_audio_profile(selected_profile)
        live_profile = LiveAudioProfile(
            name=base_profile.name,
            pre_roll_ms=microphone_settings["pre_roll_ms"],
            endpoint_grace_ms=microphone_settings["endpoint_grace_ms"],
            tail_pad_ms=microphone_settings["tail_pad_ms"],
            vad_threshold=microphone_settings["vad_threshold"],
            vad_min_silence=microphone_settings["vad_min_silence"],
            vad_min_speech=microphone_settings["vad_min_speech"],
        )
        stream_id = requested_stream_id or str(uuid.uuid4())
        vad = create_live_vad(engine.model_dir, profile=live_profile)
        vad_provider = getattr(vad, "provider_name", type(vad).__name__)
        recording = LiveRecording(RECORDINGS_DIR, stream_id)
        return LiveAudioSession(
            engine,
            vad=vad,
            pre_roll_ms=live_profile.pre_roll_ms,
            endpoint_grace_ms=live_profile.endpoint_grace_ms,
            tail_pad_ms=live_profile.tail_pad_ms,
            gate_audio=microphone_settings["vad_gating"],
            recording=recording,
        )

    try:
        if not (engine and engine.is_model_available):
            await manager.send_json(websocket, {
                "type": "error",
                "message": "X-ASR model is not loaded; live transcription is unavailable",
            })
            return

        backend_type = "X-ASR v2.0"
        await manager.send_json(websocket, {
            "type": "ready",
            "engine": backend_type,
            "protocol": "pcm_s16le/16000/mono",
            "protocol_version": 2,
            "binary_frame": "DTP2 + uint32-le sequence + pcm_s16le",
            "message": f"DiTing ready ({backend_type}); send configure then binary PCM",
        })

        if engine and engine.is_model_available:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))

                pcm_payload = message.get("bytes")
                if pcm_payload is not None:
                    if session is None:
                        raise LiveAudioProtocolError("send configure before binary PCM")
                    results = await asyncio.to_thread(session.push_binary_frame, pcm_payload)
                    for result in results:
                        await manager.send_json(websocket, {
                            "type": "live_result",
                            "data": _live_result_to_dict(result),
                        })
                    continue

                data = message.get("text")
                if data is None:
                    continue
                msg = json.loads(data)
                action = msg.get("action") or msg.get("type")

                if action in {"configure", "stream.configure"}:
                    if session is not None:
                        raise LiveAudioProtocolError("live stream is already configured")
                    sample_rate = int(msg.get("sample_rate", 16000))
                    channels = int(msg.get("channels", 1))
                    sample_format = msg.get("sample_format", "pcm_s16le")
                    if (sample_rate, channels, sample_format) != (16000, 1, "pcm_s16le"):
                        raise LiveAudioProtocolError(
                            "live stream requires 16000 Hz mono pcm_s16le"
                        )
                    session = await asyncio.to_thread(
                        build_session,
                        msg.get("profile") or msg.get("mode"),
                        msg.get("stream_id"),
                    )
                    await manager.send_json(websocket, {
                        "type": "configured",
                        "data": {
                            "stream_id": stream_id,
                            "protocol_version": 2,
                            "sample_rate": 16000,
                            "channels": 1,
                            "sample_format": "pcm_s16le",
                            "vad": vad_provider,
                            "live_profile": live_profile.name,
                            "asr_profile": engine.asr_profile,
                            "asr_chunk_ms": engine.chunk_ms,
                            "endpoint_policy": f"{vad_provider}-with-resume-grace",
                            "pre_roll_ms": live_profile.pre_roll_ms,
                            "endpoint_grace_ms": live_profile.endpoint_grace_ms,
                            "tail_pad_ms": live_profile.tail_pad_ms,
                            "vad_gating": session.metrics()["vad_gating"],
                        },
                    })
                elif action == "process_chunk":
                    # Compatibility with the former base64 Float32 JSON client.
                    if session is None:
                        session = await asyncio.to_thread(build_session)
                    audio_b64 = msg.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype="<f4")
                        pcm = (
                            np.clip(audio_chunk, -1.0, 1.0) * 32767.0
                        ).astype("<i2").tobytes()
                        results = await asyncio.to_thread(session.push_pcm_s16le, pcm)
                        for result in results:
                            await manager.send_json(websocket, {
                                "type": "live_result",
                                "data": _live_result_to_dict(result),
                            })
                elif action in {"finalize", "stop", "stream.stop"}:
                    results = await asyncio.to_thread(session.finish) if session else []
                    for result in results:
                        await manager.send_json(websocket, {
                            "type": "live_result",
                            "data": _live_result_to_dict(result),
                        })

                    canonical_error = None
                    final_enabled = runtime_config_store.load()["recognition"][
                        "final_transcription_enabled"
                    ]
                    if session and session.recording_result and final_enabled:
                        recording_result = session.recording_result
                        await manager.send_json(websocket, {
                            "type": "finalizing",
                            "data": {
                                "stream_id": stream_id,
                                "duration_ms": recording_result.duration_ms,
                                "stage": "canonical_transcription",
                            },
                        })
                        try:
                            final_engine = xasr_pool.create_final_session()
                            loop = asyncio.get_running_loop()
                            canonical_run = await loop.run_in_executor(
                                PROCESSING_EXECUTOR,
                                partial(
                                    meeting_pipeline.process_file,
                                    str(recording_result.path),
                                    final_engine,
                                    enable_diarization=True,
                                ),
                            )
                            canonical_results = canonical_run.results
                            canonical_segments = [
                                _result_to_dict(result, index + 1)
                                for index, result in enumerate(canonical_results)
                            ]
                            meeting_registry.register(
                                stream_id,
                                filename=recording_result.path.name,
                                segments=canonical_segments,
                                speakers=canonical_run.speakers,
                            )
                            await manager.send_json(websocket, {
                                "type": "final_transcript",
                                "data": {
                                    "stream_id": stream_id,
                                    "segments": canonical_segments,
                                    "segments_count": len(canonical_results),
                                    "canonical": True,
                                    "speakers": canonical_run.speakers,
                                    "diarization": canonical_run.metadata(),
                                },
                            })
                        except Exception as final_error:
                            canonical_error = str(final_error)
                            logger.error(
                                "Canonical live transcription failed for %s: %s",
                                stream_id,
                                final_error,
                            )

                    metrics = session.metrics() if session else {
                        "received_samples": 0,
                        "forwarded_samples": 0,
                    }
                    if session and session.recording_result:
                        metrics["recording"] = {
                            "stream_id": stream_id,
                            "filename": session.recording_result.path.name,
                            "duration_ms": session.recording_result.duration_ms,
                        }
                    if canonical_error:
                        metrics["canonical_error"] = canonical_error
                    await manager.send_json(websocket, {
                        "type": "stopped",
                        "data": {**metrics, "vad": vad_provider},
                    })
                    normal_stop = True
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Live WS error: {e}")
        try:
            await manager.send_json(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        manager.disconnect(websocket)
        if session is not None and not normal_stop:
            try:
                recovery_path = await asyncio.to_thread(session.abort)
                if recovery_path:
                    logger.warning(
                        "Live stream %s disconnected; recoverable recording kept at %s",
                        stream_id,
                        recovery_path,
                    )
            except Exception as close_error:
                logger.warning(f"Live session cleanup failed: {close_error}")
        elif engine:
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
