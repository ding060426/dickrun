"""Persistent recognition and microphone settings for new ASR sessions."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path


ASR_PROFILES = ("low-latency", "balanced", "meeting", "quality")
ASR_PROVIDERS = ("xasr", "qwen3")
QWEN3_DEVICES = ("auto", "cuda:0", "cpu")
QWEN3_DTYPES = ("auto", "bfloat16", "float16", "float32")

DEFAULT_RUNTIME_SETTINGS = {
    "recognition": {
        "asr_provider": "xasr",
        "qwen3_model_path": "",
        "qwen3_device": "auto",
        "qwen3_dtype": "auto",
        "live_asr_profile": "meeting",
        "final_asr_profile": "meeting",
        "final_transcription_enabled": True,
        "file_vad_provider": "silero",
        "file_vad_threshold": 0.5,
        "file_vad_min_silence": 0.5,
        "file_vad_min_speech": 0.2,
        "file_vad_pre_padding_ms": 250,
        "file_vad_post_padding_ms": 450,
    },
    "microphone": {
        "device_id": "",
        "live_profile": "meeting",
        "vad_gating": False,
        "echo_cancellation": True,
        "noise_suppression": False,
        "auto_gain_control": False,
        "pre_roll_ms": 700,
        "endpoint_grace_ms": 800,
        "tail_pad_ms": 1000,
        "vad_threshold": 0.5,
        "vad_min_silence": 0.5,
        "vad_min_speech": 0.2,
    },
}


def _number(value, default, minimum, maximum, *, integer=False):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = min(float(maximum), max(float(minimum), parsed))
    return int(round(parsed)) if integer else round(parsed, 3)


def _choice(value, choices, default):
    selected = str(value or "").strip().lower()
    return selected if selected in choices else default


class RuntimeConfigStore:
    """Load, validate, and atomically save runtime settings."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(self) -> dict:
        with self._lock:
            if self.path.is_file():
                try:
                    return self.normalize(json.loads(self.path.read_text(encoding="utf-8")))
                except (OSError, TypeError, ValueError):
                    pass
            return self.normalize({})

    def save(self, payload: dict) -> dict:
        normalized = self.normalize(payload)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        return normalized

    def normalize(self, payload: dict | None) -> dict:
        source = payload if isinstance(payload, dict) else {}
        recognition_source = source.get("recognition", {})
        microphone_source = source.get("microphone", {})
        if not isinstance(recognition_source, dict):
            recognition_source = {}
        if not isinstance(microphone_source, dict):
            microphone_source = {}

        defaults = deepcopy(DEFAULT_RUNTIME_SETTINGS)
        recognition_default = defaults["recognition"]
        microphone_default = defaults["microphone"]

        recognition = {
            "asr_provider": _choice(
                recognition_source.get("asr_provider"),
                ASR_PROVIDERS,
                recognition_default["asr_provider"],
            ),
            "qwen3_model_path": str(
                recognition_source.get("qwen3_model_path", "") or ""
            ).strip()[:1024],
            "qwen3_device": _choice(
                recognition_source.get("qwen3_device"),
                QWEN3_DEVICES,
                recognition_default["qwen3_device"],
            ),
            "qwen3_dtype": _choice(
                recognition_source.get("qwen3_dtype"),
                QWEN3_DTYPES,
                recognition_default["qwen3_dtype"],
            ),
            "live_asr_profile": _choice(
                recognition_source.get("live_asr_profile"),
                ASR_PROFILES,
                recognition_default["live_asr_profile"],
            ),
            "final_asr_profile": _choice(
                recognition_source.get("final_asr_profile"),
                ASR_PROFILES,
                recognition_default["final_asr_profile"],
            ),
            "final_transcription_enabled": bool(
                recognition_source.get(
                    "final_transcription_enabled",
                    recognition_default["final_transcription_enabled"],
                )
            ),
            "file_vad_provider": "silero",
            "file_vad_threshold": _number(
                recognition_source.get("file_vad_threshold"), 0.5, 0.05, 0.95
            ),
            "file_vad_min_silence": _number(
                recognition_source.get("file_vad_min_silence"), 0.5, 0.1, 3.0
            ),
            "file_vad_min_speech": _number(
                recognition_source.get("file_vad_min_speech"), 0.2, 0.05, 2.0
            ),
            "file_vad_pre_padding_ms": _number(
                recognition_source.get("file_vad_pre_padding_ms"), 250, 0, 2000, integer=True
            ),
            "file_vad_post_padding_ms": _number(
                recognition_source.get("file_vad_post_padding_ms"), 450, 0, 3000, integer=True
            ),
        }
        microphone = {
            "device_id": str(microphone_source.get("device_id", ""))[:512],
            "live_profile": _choice(
                microphone_source.get("live_profile"),
                ("meeting", "dictation", "oncall"),
                microphone_default["live_profile"],
            ),
            "vad_gating": bool(microphone_source.get("vad_gating", False)),
            "echo_cancellation": bool(microphone_source.get("echo_cancellation", True)),
            "noise_suppression": bool(microphone_source.get("noise_suppression", False)),
            "auto_gain_control": bool(microphone_source.get("auto_gain_control", False)),
            "pre_roll_ms": _number(
                microphone_source.get("pre_roll_ms"), 700, 0, 3000, integer=True
            ),
            "endpoint_grace_ms": _number(
                microphone_source.get("endpoint_grace_ms"), 800, 0, 5000, integer=True
            ),
            "tail_pad_ms": _number(
                microphone_source.get("tail_pad_ms"), 1000, 0, 3000, integer=True
            ),
            "vad_threshold": _number(
                microphone_source.get("vad_threshold"), 0.5, 0.05, 0.95
            ),
            "vad_min_silence": _number(
                microphone_source.get("vad_min_silence"), 0.5, 0.1, 3.0
            ),
            "vad_min_speech": _number(
                microphone_source.get("vad_min_speech"), 0.2, 0.05, 2.0
            ),
        }
        return {"recognition": recognition, "microphone": microphone}
