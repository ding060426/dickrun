"""Atomically managed live/final X-ASR runtimes with profile fallback."""

from __future__ import annotations

import threading
from pathlib import Path

from .asr_engine import XASREngine
from .config import ASR_CHUNK_PROFILES


PROFILE_FALLBACK_ORDER = ("meeting", "balanced", "low-latency", "quality")


def deployed_profiles(model_dir: str | Path) -> list[str]:
    root = Path(model_dir)
    if not (root / "tokens.txt").is_file():
        return []
    return [
        profile
        for profile, chunk_ms in ASR_CHUNK_PROFILES.items()
        if all(
            (root / f"{component}-{chunk_ms}ms.onnx").is_file()
            for component in ("encoder", "decoder", "joiner")
        )
    ]


def resolve_deployed_profile(requested: str, available: list[str]) -> str | None:
    if requested in available:
        return requested
    return next((profile for profile in PROFILE_FALLBACK_ORDER if profile in available), None)


class AsrEnginePool:
    """Own the process-level live and canonical recognizer runtimes."""

    def __init__(self, model_dir: str | Path, *, engine_factory=XASREngine, base_options=None):
        self.model_dir = Path(model_dir)
        self.engine_factory = engine_factory
        self.base_options = dict(base_options or {})
        self._lock = threading.RLock()
        self.live_engine = None
        self.final_engine = None
        self._status = self._empty_status()

    def reload(self, recognition: dict, hotwords: dict) -> dict:
        available = deployed_profiles(self.model_dir)
        requested_live = str(recognition.get("live_asr_profile", "meeting"))
        requested_final = str(recognition.get("final_asr_profile", "meeting"))
        effective_live = resolve_deployed_profile(requested_live, available)
        effective_final = resolve_deployed_profile(requested_final, available)
        if effective_live is None:
            raise FileNotFoundError(f"No complete X-ASR profile found in {self.model_dir}")
        if effective_final is None:
            effective_final = effective_live

        common = self._engine_options(recognition, hotwords)
        live = self.engine_factory(asr_profile=effective_live, **common).warmup()
        final = live
        if effective_final != effective_live:
            final = self.engine_factory(asr_profile=effective_final, **common).warmup()

        status = {
            "available_profiles": available,
            "live": self._profile_status(requested_live, effective_live),
            "final": self._profile_status(requested_final, effective_final),
            "shared_runtime": live is final,
            "file_vad_provider": getattr(final, "file_vad_provider", "silero"),
        }
        with self._lock:
            self.live_engine = live
            self.final_engine = final
            self._status = status
        return self.status()

    def create_live_session(self):
        with self._lock:
            engine = self.live_engine
        return engine.fork_session() if engine is not None else None

    def create_final_session(self):
        with self._lock:
            engine = self.final_engine
        return engine.fork_session() if engine is not None else None

    def configure_hotwords(self, hotwords: dict) -> None:
        words, scores = self._hotword_inputs(hotwords)
        with self._lock:
            engines = {id(engine): engine for engine in (self.live_engine, self.final_engine) if engine}
        for engine in engines.values():
            engine.configure_hotwords(
                words,
                scores=scores,
                default_score=hotwords.get("default_score", 5.0),
                enabled=hotwords.get("enabled", True),
                fuzzy_pinyin_enabled=hotwords.get("fuzzy_pinyin_enabled", True),
            )

    def status(self) -> dict:
        with self._lock:
            return {
                **self._status,
                "available_profiles": list(self._status.get("available_profiles", [])),
                "live": dict(self._status.get("live", {})),
                "final": dict(self._status.get("final", {})),
            }

    def _engine_options(self, recognition: dict, hotwords: dict) -> dict:
        words, scores = self._hotword_inputs(hotwords)
        file_vad_options = {
            "threshold": recognition.get("file_vad_threshold", 0.5),
            "min_silence_duration": recognition.get("file_vad_min_silence", 0.5),
            "min_speech_duration": recognition.get("file_vad_min_speech", 0.2),
            "pre_padding_ms": recognition.get("file_vad_pre_padding_ms", 250),
            "post_padding_ms": recognition.get("file_vad_post_padding_ms", 450),
        }
        return {
            **self.base_options,
            "model_dir": str(self.model_dir),
            "hotwords": words,
            "hotword_scores": scores,
            "hotwords_score": hotwords.get("default_score", 5.0),
            "enable_hotword_correction": hotwords.get("enabled", True),
            "enable_fuzzy_pinyin": hotwords.get("fuzzy_pinyin_enabled", True),
            "file_vad_options": file_vad_options,
        }

    @staticmethod
    def _hotword_inputs(hotwords: dict) -> tuple[list[str], dict[str, float]]:
        if not hotwords.get("enabled", True):
            return [], {}
        active = [item for item in hotwords.get("words", []) if item.get("enabled", True)]
        return (
            [str(item.get("text", "")).strip() for item in active if str(item.get("text", "")).strip()],
            {
                str(item.get("text", "")).strip(): float(item.get("score", hotwords.get("default_score", 5.0)))
                for item in active
                if str(item.get("text", "")).strip()
            },
        )

    @staticmethod
    def _profile_status(requested: str, effective: str) -> dict:
        return {
            "requested_profile": requested,
            "effective_profile": effective,
            "chunk_ms": ASR_CHUNK_PROFILES[effective],
            "fallback": requested != effective,
        }

    @staticmethod
    def _empty_status() -> dict:
        return {
            "available_profiles": [],
            "live": {},
            "final": {},
            "shared_runtime": False,
            "file_vad_provider": "unavailable",
        }
