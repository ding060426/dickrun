"""Optional Qwen3-ASR adapter for file and canonical transcription."""

from __future__ import annotations

import gc
import importlib.util
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Callable

try:
    import numpy as np
except ImportError:  # Optional until Qwen3 is actually used.
    np = None

from audio_buffer import AudioBuffer, load_audio_buffer

from .contracts import ASRResult


class Qwen3AsrEngine:
    """Share one lazily loaded Qwen3 model across lightweight file sessions."""

    provider_name = "qwen3"
    engine_name = "Qwen3-ASR"
    file_vad_provider = "qwen3-full-audio"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        hotwords: list[str] | None = None,
        hotword_scores: dict[str, float] | None = None,
        enable_hotword_correction: bool = True,
        num_threads: int = 12,
        model_loader: Callable | None = None,
        torch_module=None,
        shared_runtime: dict | None = None,
        **_ignored,
    ):
        self.model_path = str(model_path or "").strip()
        self.device = str(device or "auto").lower()
        self.dtype = str(dtype or "auto").lower()
        self.num_threads = max(1, int(num_threads))
        self.logic_validator = None
        self.is_model_available = False
        self._model_loader = model_loader
        self._torch = torch_module
        self._runtime = shared_runtime or {"model": None, "lock": threading.RLock()}
        self._runtime.setdefault("owners", weakref.WeakSet()).add(self)
        self._runtime.setdefault("last_used_at", None)
        self._runtime.setdefault("device", self.device)
        self._runtime.setdefault("dtype", self.dtype)
        self._runtime.setdefault("last_error", "")
        self.idle_timeout_sec = max(0, int(os.getenv("DITING_QWEN3_IDLE_UNLOAD_SEC", "0") or "0"))
        self._hotwords = list(hotwords or []) if enable_hotword_correction else []
        self._hotword_scores = dict(hotword_scores or {})

    @classmethod
    def availability(cls, model_path: str = "") -> dict:
        if not str(model_path or "").strip():
            return {"available": False, "reason": "model_not_configured"}
        missing = [
            package
            for package in ("torch", "qwen_asr")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            return {
                "available": False,
                "reason": "missing_dependencies",
                "missing": missing,
            }
        return {"available": True, "reason": ""}

    def warmup(self):
        if self._runtime["model"] is not None:
            self.is_model_available = True
            return self
        availability = self.availability(self.model_path)
        if self._model_loader is None and not availability["available"]:
            raise RuntimeError(f"Qwen3-ASR unavailable: {availability['reason']}")
        with self._runtime["lock"]:
            if self._runtime["model"] is None:
                torch = self._torch
                if torch is None:
                    import torch as torch_module

                    torch = torch_module
                    self._torch = torch
                set_num_threads = getattr(torch, "set_num_threads", None)
                if callable(set_num_threads):
                    set_num_threads(self.num_threads)
                loader = self._model_loader
                if loader is None:
                    from qwen_asr import Qwen3ASRModel

                    loader = Qwen3ASRModel.from_pretrained
                device = self._resolve_device(torch)
                dtype = self._resolve_dtype(torch, device)
                self._runtime["device"] = device
                self._runtime["dtype"] = self.dtype if self.dtype != "auto" else ("bfloat16" if device.startswith("cuda") else "float32")
                self._runtime["last_error"] = ""
                self._runtime["model"] = loader(
                    self.model_path,
                    dtype=dtype,
                    device_map=device,
                    max_inference_batch_size=1,
                    max_new_tokens=1024,
                )
        self.is_model_available = True
        self._runtime["last_used_at"] = time.time()
        return self

    def fork_session(self):
        session = self.__class__(
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            num_threads=self.num_threads,
            hotwords=self._hotwords,
            hotword_scores=self._hotword_scores,
            model_loader=self._model_loader,
            torch_module=self._torch,
            shared_runtime=self._runtime,
        )
        session.is_model_available = self.is_model_available
        return session

    def close(self) -> None:
        """Drop the shared model and return cached CUDA memory to the driver."""
        model = None
        with self._runtime["lock"]:
            model = self._runtime.get("model")
            self._runtime["model"] = None
            self._runtime["last_used_at"] = None
            for owner in tuple(self._runtime.get("owners", ())):
                owner.is_model_available = False
        if model is None:
            return
        del model
        gc.collect()
        self._release_cuda_cache()

    def configure_hotwords(
        self,
        words,
        *,
        scores=None,
        enabled=True,
        **_ignored,
    ) -> None:
        self._hotwords = list(words or []) if enabled else []
        self._hotword_scores = dict(scores or {})

    def process_file(
        self,
        file_path: str,
        on_segment=None,
        on_progress=None,
        audio_buffer: AudioBuffer | None = None,
    ) -> list[ASRResult]:
        if on_progress:
            on_progress("loading", 0.0)
        audio = audio_buffer or load_audio_buffer(file_path)
        if not len(audio.samples):
            if on_progress:
                on_progress("done", 1.0)
            return []
        if on_progress:
            on_progress("recognizing", 0.2)
        result = self._transcribe(audio.samples, audio.sample_rate, 0.0, audio.duration)
        results = [result] if result and result.text else []
        if result and on_segment:
            on_segment(result, 1, 1)
        if on_progress:
            on_progress("done", 1.0)
        return results

    def recognize_interval(
        self,
        audio_buffer: AudioBuffer,
        start_sec: float,
        end_sec: float,
        *,
        pre_padding_ms: int = 250,
        post_padding_ms: int = 400,
    ) -> ASRResult | None:
        start_sec = max(0.0, float(start_sec))
        end_sec = min(audio_buffer.duration, float(end_sec))
        if end_sec <= start_sec:
            return None
        decode_start = max(0.0, start_sec - max(0, pre_padding_ms) / 1000)
        decode_end = min(audio_buffer.duration, end_sec + max(0, post_padding_ms) / 1000)
        result = self._transcribe(
            audio_buffer.slice(decode_start, decode_end),
            audio_buffer.sample_rate,
            start_sec,
            end_sec,
        )
        if result:
            result.audio_data = audio_buffer.slice(start_sec, end_sec)
        return result

    def _transcribe(self, samples, sample_rate: int, start: float, end: float):
        if np is None:
            raise RuntimeError("Qwen3-ASR requires numpy")
        if self._runtime["model"] is None:
            self.warmup()
        context = self._hotword_context()
        kwargs = {"audio": (np.asarray(samples, dtype=np.float32), sample_rate), "language": None}
        if context:
            kwargs["context"] = context
        try:
            with self._runtime["lock"]:
                output = self._runtime["model"].transcribe(**kwargs)
                self._runtime["last_used_at"] = time.time()
                self._runtime["last_error"] = ""
        except Exception as exc:
            self._runtime["last_error"] = str(exc)
            if self._is_cuda_oom(exc):
                self.close()
            raise
        item = output[0] if isinstance(output, (list, tuple)) and output else output
        text = str(getattr(item, "text", item if isinstance(item, str) else "") or "").strip()
        if not text:
            return None
        return ASRResult(
            text=text,
            raw_text=text,
            is_final=True,
            timestamp=end,
            start_sec=start,
            end_sec=end,
            audio_data=np.asarray(samples, dtype=np.float32),
        )

    def _hotword_context(self) -> str:
        if not self._hotwords:
            return ""
        ordered = sorted(
            self._hotwords,
            key=lambda word: self._hotword_scores.get(word, 0.0),
            reverse=True,
        )
        return "可能出现的专业词汇：" + "、".join(ordered[:100])

    def _resolve_device(self, torch):
        if self.device != "auto":
            return self.device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch, device: str):
        selected = self.dtype
        if selected == "auto":
            selected = "bfloat16" if device.startswith("cuda") else "float32"
        return getattr(torch, selected)

    def _is_cuda_oom(self, exc: Exception) -> bool:
        torch = self._torch
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        oom_type = getattr(cuda, "OutOfMemoryError", None)
        return bool(
            (oom_type is not None and isinstance(exc, oom_type))
            or "out of memory" in str(exc).lower()
        )

    def runtime_status(self) -> dict:
        last_used_at = self._runtime.get("last_used_at")
        return {
            "loaded": self._runtime.get("model") is not None,
            "last_used_at": last_used_at,
            "idle_timeout_sec": self.idle_timeout_sec,
            "device": self._runtime.get("device", self.device),
            "dtype": self._runtime.get("dtype", self.dtype),
            "estimated_vram_gb": 5.5,
            "last_error": self._runtime.get("last_error", ""),
        }

    def maybe_unload_idle(self, now: float | None = None) -> bool:
        if self.idle_timeout_sec <= 0 or self._runtime.get("model") is None:
            return False
        last_used_at = self._runtime.get("last_used_at")
        if not last_used_at:
            return False
        current = time.time() if now is None else float(now)
        if current - float(last_used_at) < self.idle_timeout_sec:
            return False
        self.close()
        return True

    def _release_cuda_cache(self) -> None:
        torch = self._torch
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is None or not cuda.is_available():
            return
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except RuntimeError:
                pass
