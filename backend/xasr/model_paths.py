"""Centralized path resolution for local Huiwu model and runtime assets."""

from __future__ import annotations

import os
from pathlib import Path

from .config import ASR_CHUNK_PROFILES


DIARIZATION_SEGMENTATION_FILENAME = "pyannote-segmentation-3.0.int8.onnx"
DIARIZATION_EMBEDDING_FILENAME = "3dspeaker-eres2net.onnx"


def _env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def resolve_project_root() -> Path:
    """Return the repository root inferred from backend/xasr/."""

    return Path(__file__).resolve().parents[2]


def resolve_backend_dir() -> Path:
    return resolve_project_root() / "backend"


def resolve_models_root() -> Path:
    return _env_path("HUIWU_MODELS_DIR") or (resolve_project_root() / "models")


def resolve_xasr_model_dir() -> Path:
    return _env_path("DITING_XASR_MODEL_DIR") or (resolve_models_root() / "xasr")


def resolve_qwen3_model_dir() -> Path:
    return _env_path("DITING_QWEN3_MODEL_PATH") or (resolve_models_root() / "qwen3")


def resolve_vad_model_path() -> Path:
    return _env_path("DITING_SILERO_VAD_PATH") or (resolve_models_root() / "vad" / "silero_vad.onnx")


def resolve_diarization_model_dir() -> Path:
    return _env_path("DITING_DIARIZATION_MODEL_DIR") or (resolve_models_root() / "diarization")


def resolve_diarization_segmentation_model() -> Path:
    return (
        _env_path("DITING_DIARIZATION_SEGMENTATION_MODEL")
        or resolve_diarization_model_dir() / DIARIZATION_SEGMENTATION_FILENAME
    )


def resolve_speaker_embedding_model() -> Path:
    return (
        _env_path("DITING_SPEAKER_EMBEDDING_MODEL")
        or resolve_diarization_model_dir() / DIARIZATION_EMBEDDING_FILENAME
    )


def resolve_recordings_dir() -> Path:
    return _env_path("DITING_RECORDINGS_DIR") or (resolve_backend_dir() / "recordings")


def resolve_runtime_config_path() -> Path:
    return _env_path("DITING_RUNTIME_CONFIG") or (resolve_backend_dir() / "data" / "settings.json")


def resolve_hotwords_config_path() -> Path:
    return _env_path("DITING_HOTWORDS_CONFIG") or (resolve_backend_dir() / "data" / "hotwords.json")


def xasr_profile_files(profile: str, model_dir: str | Path | None = None) -> dict[str, Path]:
    chunk_ms = ASR_CHUNK_PROFILES[profile]
    root = Path(model_dir) if model_dir is not None else resolve_xasr_model_dir()
    return {
        "tokens": root / "tokens.txt",
        "encoder": root / f"encoder-{chunk_ms}ms.onnx",
        "decoder": root / f"decoder-{chunk_ms}ms.onnx",
        "joiner": root / f"joiner-{chunk_ms}ms.onnx",
    }


def inspect_xasr_profiles(model_dir: str | Path | None = None) -> dict[str, dict]:
    root = Path(model_dir) if model_dir is not None else resolve_xasr_model_dir()
    profiles: dict[str, dict] = {}
    for profile, chunk_ms in ASR_CHUNK_PROFILES.items():
        files = xasr_profile_files(profile, root)
        missing = [name for name, path in files.items() if not path.is_file()]
        profiles[profile] = {
            "profile": profile,
            "chunk_ms": chunk_ms,
            "complete": not missing,
            "missing": missing,
            "files": {name: str(path) for name, path in files.items()},
        }
    return profiles


def inspect_model_paths() -> dict:
    profiles = inspect_xasr_profiles()
    qwen3_dir = resolve_qwen3_model_dir()
    qwen3_required = ["config.json"]
    qwen3_safetensors = list(qwen3_dir.glob("*.safetensors")) if qwen3_dir.is_dir() else []
    qwen3_missing = [name for name in qwen3_required if not (qwen3_dir / name).is_file()]
    if not qwen3_safetensors:
        qwen3_missing.append("*.safetensors")

    segmentation = resolve_diarization_segmentation_model()
    embedding = resolve_speaker_embedding_model()
    vad = resolve_vad_model_path()
    return {
        "project_root": str(resolve_project_root()),
        "backend_dir": str(resolve_backend_dir()),
        "models_root": str(resolve_models_root()),
        "xasr_model_dir": str(resolve_xasr_model_dir()),
        "qwen3_model_dir": str(qwen3_dir),
        "vad_model_path": str(vad),
        "diarization_model_dir": str(resolve_diarization_model_dir()),
        "diarization_segmentation_model": str(segmentation),
        "speaker_embedding_model": str(embedding),
        "recordings_dir": str(resolve_recordings_dir()),
        "runtime_config_path": str(resolve_runtime_config_path()),
        "hotwords_config_path": str(resolve_hotwords_config_path()),
        "xasr_profiles": profiles,
        "vad": {
            "available": vad.is_file(),
            "missing": [] if vad.is_file() else [str(vad)],
        },
        "qwen3": {
            "available": qwen3_dir.is_dir() and not qwen3_missing,
            "missing": qwen3_missing,
            "safetensors": [str(path) for path in qwen3_safetensors],
        },
        "diarization": {
            "available": segmentation.is_file() and embedding.is_file(),
            "missing": [
                str(path)
                for path in (segmentation, embedding)
                if not path.is_file()
            ],
        },
    }
