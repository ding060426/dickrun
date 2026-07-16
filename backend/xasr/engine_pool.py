"""Atomically managed live/final X-ASR runtimes with profile fallback."""

from __future__ import annotations

import threading
from pathlib import Path

from .asr_engine import XASREngine
from .config import ASR_CHUNK_PROFILES
from .qwen3_engine import Qwen3AsrEngine


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

    def __init__(
        self,
        model_dir: str | Path,
        *,
        engine_factory=XASREngine,
        qwen_engine_factory=Qwen3AsrEngine,
        base_options=None,
    ):
        self.model_dir = Path(model_dir)
        self.engine_factory = engine_factory
        self.qwen_engine_factory = qwen_engine_factory
        self.base_options = dict(base_options or {})
        self._lock = threading.RLock()
        self.live_engine = None
        self.final_engine = None
        self._status = self._empty_status()

    def reload(self, recognition: dict, hotwords: dict) -> dict:
        with self._lock:
            previous_engines = self._unique_engines(
                self.live_engine,
                self.final_engine,
            )
            previous_qwen_ids = {
                id(engine)
                for engine in previous_engines
                if getattr(engine, "provider_name", "") == "qwen3"
            }
            if self.final_engine is not None and id(self.final_engine) in previous_qwen_ids:
                self.final_engine = self.live_engine
                self._status = {
                    **self._status,
                    "effective_provider": "xasr",
                    "provider_fallback": True,
                    "provider_reason": "runtime_reloading",
                    "shared_runtime": True,
                    "file_vad_provider": getattr(
                        self.live_engine,
                        "file_vad_provider",
                        "silero",
                    ),
                }
        preclosed_ids = {
            id(engine)
            for engine in previous_engines
            if id(engine) in previous_qwen_ids and self._close_engine(engine)
        }
        available = deployed_profiles(self.model_dir)
        requested_live = str(recognition.get("live_asr_profile", "meeting"))
        requested_final = str(recognition.get("final_asr_profile", "meeting"))
        selected_provider = str(recognition.get("asr_provider", "xasr")).lower()
        effective_live = resolve_deployed_profile(requested_live, available)
        effective_final = resolve_deployed_profile(requested_final, available)
        if effective_live is None:
            raise FileNotFoundError(f"No complete X-ASR profile found in {self.model_dir}")
        if effective_final is None:
            effective_final = effective_live

        common = self._engine_options(recognition, hotwords)
        live = self.engine_factory(asr_profile=effective_live, **common).warmup()
        xasr_final = live
        effective_xasr_final = effective_live

        def ensure_xasr_final():
            nonlocal xasr_final, effective_xasr_final
            if effective_final != effective_live and xasr_final is live:
                xasr_final = self.engine_factory(
                    asr_profile=effective_final,
                    **common,
                ).warmup()
                effective_xasr_final = effective_final
            return xasr_final

        qwen_availability = self.qwen_engine_factory.availability(
            recognition.get("qwen3_model_path", "")
        )
        final = live
        effective_provider = "xasr"
        provider_reason = ""
        if selected_provider == "qwen3":
            if qwen_availability.get("available"):
                try:
                    qwen_options = self._qwen_options(recognition, hotwords)
                    final = self.qwen_engine_factory(**qwen_options).warmup()
                    effective_provider = "qwen3"
                except Exception as exc:
                    qwen_availability = {
                        "available": False,
                        "reason": "initialization_failed",
                        "detail": str(exc),
                    }
                    provider_reason = "initialization_failed"
            else:
                provider_reason = str(qwen_availability.get("reason", "unavailable"))
            if effective_provider != "qwen3":
                final = ensure_xasr_final()
        else:
            final = ensure_xasr_final()

        status = {
            "available_profiles": available,
            "selected_provider": selected_provider,
            "effective_provider": effective_provider,
            "live_provider": "xasr",
            "provider_fallback": selected_provider != effective_provider,
            "provider_reason": provider_reason,
            "providers": {
                "xasr": {"available": True, "reason": ""},
                "qwen3": dict(qwen_availability),
            },
            "live": self._profile_status(requested_live, effective_live),
            "final": self._profile_status(requested_final, effective_xasr_final),
            "shared_runtime": live is final,
            "file_vad_provider": getattr(final, "file_vad_provider", "silero"),
            "inference_threads": max(1, int(common.get("num_threads", 1))),
        }
        with self._lock:
            self.live_engine = live
            self.final_engine = final
            self._status = status
        active_ids = {id(engine) for engine in self._unique_engines(live, final)}
        for engine in previous_engines:
            if id(engine) not in active_ids and id(engine) not in preclosed_ids:
                self._close_engine(engine)
        return self.status()

    def close(self) -> None:
        """Release recognizer runtimes and any model-owned accelerator cache."""
        with self._lock:
            engines = self._unique_engines(self.live_engine, self.final_engine)
            self.live_engine = None
            self.final_engine = None
            self._status = self._empty_status()
        for engine in engines:
            self._close_engine(engine)

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
                "providers": {
                    name: dict(details)
                    for name, details in self._status.get("providers", {}).items()
                },
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

    def _qwen_options(self, recognition: dict, hotwords: dict) -> dict:
        words, scores = self._hotword_inputs(hotwords)
        return {
            "model_path": recognition.get("qwen3_model_path", ""),
            "device": recognition.get("qwen3_device", "auto"),
            "dtype": recognition.get("qwen3_dtype", "auto"),
            "num_threads": max(1, int(self.base_options.get("num_threads", 12))),
            "hotwords": words,
            "hotword_scores": scores,
            "enable_hotword_correction": hotwords.get("enabled", True),
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
    def _unique_engines(*engines) -> list:
        return list({id(engine): engine for engine in engines if engine}.values())

    @staticmethod
    def _close_engine(engine) -> bool:
        close = getattr(engine, "close", None)
        if not callable(close):
            return False
        close()
        return True

    def _empty_status(self) -> dict:
        return {
            "available_profiles": [],
            "selected_provider": "xasr",
            "effective_provider": "xasr",
            "live_provider": "xasr",
            "provider_fallback": False,
            "provider_reason": "",
            "providers": {
                "xasr": {"available": False, "reason": "not_loaded"},
                "qwen3": Qwen3AsrEngine.availability(""),
            },
            "live": {},
            "final": {},
            "shared_runtime": False,
            "file_vad_provider": "unavailable",
            "inference_threads": max(1, int(self.base_options.get("num_threads", 1))),
        }
