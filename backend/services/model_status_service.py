"""Build model/runtime status payloads for diagnostics and settings UI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from xasr.model_paths import inspect_model_paths


QWEN3_ESTIMATED_VRAM_GB = 5.5


def _dependency_status(*packages: str) -> dict[str, Any]:
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    return {
        "available": not missing,
        "missing": missing,
    }


def _provider_status(pool_status: dict, runtime_settings: dict) -> dict:
    recognition = runtime_settings.get("recognition", {})
    qwen3_provider = dict(pool_status.get("providers", {}).get("qwen3", {}))
    qwen3_dependencies = _dependency_status("torch", "qwen_asr")
    qwen3_provider.setdefault("available", qwen3_dependencies["available"])
    qwen3_provider.setdefault(
        "reason",
        "" if qwen3_dependencies["available"] else "missing_dependencies",
    )
    if qwen3_dependencies["missing"]:
        qwen3_provider.setdefault("missing", qwen3_dependencies["missing"])
    qwen3_provider.update(
        {
            "configured_path": recognition.get("qwen3_model_path", ""),
            "dependencies": qwen3_dependencies,
            "loaded": bool(qwen3_provider.get("loaded", False)),
            "device": recognition.get("qwen3_device", "auto"),
            "dtype": recognition.get("qwen3_dtype", "auto"),
            "estimated_vram_gb": QWEN3_ESTIMATED_VRAM_GB,
            "mode": "final_transcription_only",
        }
    )
    return {
        "selected_provider": pool_status.get("selected_provider", recognition.get("asr_provider", "xasr")),
        "effective_provider": pool_status.get("effective_provider", "xasr"),
        "live_provider": pool_status.get("live_provider", "xasr"),
        "provider_fallback": bool(pool_status.get("provider_fallback", False)),
        "provider_reason": pool_status.get("provider_reason", ""),
        "xasr": dict(pool_status.get("providers", {}).get("xasr", {"available": True, "reason": ""})),
        "qwen3": qwen3_provider,
    }


def _profile_summary(pool_status: dict, model_paths: dict) -> dict:
    return {
        "available_profiles": list(pool_status.get("available_profiles", [])),
        "live": dict(pool_status.get("live", {})),
        "final": dict(pool_status.get("final", {})),
        "xasr_profiles": model_paths.get("xasr_profiles", {}),
    }


def _diarization_status(meeting_pipeline) -> dict:
    status = meeting_pipeline.status() if meeting_pipeline is not None else {"available": False}
    model_paths = inspect_model_paths()
    missing = model_paths["diarization"]["missing"]
    return {
        **status,
        "mode": "diarization" if status.get("available") else "asr_only",
        "available": bool(status.get("available")),
        "reason": "" if status.get("available") else "missing_models",
        "required_models": [
            model_paths["diarization_segmentation_model"],
            model_paths["speaker_embedding_model"],
        ],
        "missing_models": missing,
        "model_dir": model_paths["diarization_model_dir"],
    }


def build_xasr_status(
    *,
    has_xasr: bool,
    xasr_engine,
    xasr_pool,
    xasr_loading: bool,
    runtime_settings: dict,
    hotword_settings: dict,
    meeting_pipeline,
    live_vad_model,
    processing_workers: int = 0,
    active_live_sessions: int = 0,
    upload_jobs: dict | None = None,
) -> dict:
    """Return a backward-compatible /api/xasr/status payload."""

    model_paths = inspect_model_paths()
    pool_status = xasr_pool.status() if xasr_pool is not None else {}
    recognition = runtime_settings.get("recognition", {})
    microphone = runtime_settings.get("microphone", {})
    upload_jobs = upload_jobs or {"queued": 0, "running": 0}

    base = {
        "available": bool(has_xasr),
        "model_available": bool(getattr(xasr_engine, "is_model_available", False)),
        "model_dir": str(getattr(xasr_engine, "model_dir", model_paths["xasr_model_dir"])),
        "endpoint_detection": bool(getattr(xasr_engine, "enable_endpoint_detection", False)),
        "loading": bool(xasr_loading),
        "paths": {
            "project_root": model_paths["project_root"],
            "models_root": model_paths["models_root"],
            "xasr_model_dir": model_paths["xasr_model_dir"],
            "vad_model_path": model_paths["vad_model_path"],
            "qwen3_model_dir": model_paths["qwen3_model_dir"],
            "diarization_model_dir": model_paths["diarization_model_dir"],
        },
        "profiles": _profile_summary(pool_status, model_paths),
        "providers": _provider_status(pool_status, runtime_settings),
        "models": pool_status,
        "file_vad": {
            "provider": pool_status.get("file_vad_provider", "unavailable"),
            "available": model_paths["vad"]["available"],
            "model_path": model_paths["vad_model_path"],
            "threshold": recognition.get("file_vad_threshold", 0.5),
            "min_silence": recognition.get("file_vad_min_silence", 0.5),
            "min_speech": recognition.get("file_vad_min_speech", 0.2),
            "missing_reason": "" if model_paths["vad"]["available"] else "model_missing",
        },
        "diarization": _diarization_status(meeting_pipeline),
        "hotwords_count": int(hotword_settings.get("active_count", 0)),
        "resources": {
            "inference_threads": pool_status.get("inference_threads", 0),
            "processing_workers": processing_workers,
            "active_live_sessions": active_live_sessions,
            "upload_jobs": upload_jobs,
        },
    }
    base["live_vad"] = {
        "provider": "sherpa-silero-vad" if live_vad_model else "asr-endpoint-fallback",
        "available": live_vad_model is not None,
        "model_path": str(live_vad_model) if live_vad_model else None,
        "endpoint_policy": "silero-vad-with-resume-grace" if live_vad_model else "sherpa-asr-endpoint-fallback",
        "endpoint_grace_ms": microphone.get("endpoint_grace_ms", 0) if live_vad_model else 0,
    }
    features = {
        "logic_validation": bool(getattr(xasr_engine, "enable_logic_validation", False)),
        "hotword_correction": bool(getattr(xasr_engine, "enable_hotword_correction", False)),
        "fuzzy_pinyin": bool(getattr(xasr_engine, "enable_fuzzy_pinyin", False)),
        "uncertainty_estimation": bool(getattr(xasr_engine, "enable_uncertainty", False)),
        "speaker_diarization": bool(base["diarization"].get("available")),
    }
    base["features"] = features
    if not has_xasr:
        base["reason"] = "X-ASR module not installed"
    elif xasr_engine is None:
        base["reason"] = "Loading model..." if xasr_loading else "Engine init failed"
    return base
