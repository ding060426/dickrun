"""Optional RNNoise wrapper for DiTing/HUIWU.

RNNoise is a native C library. This module loads a prebuilt rnnoise.dll when
available and otherwise transparently returns the input audio unchanged.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from modules.pcm_utils import safe_resample, sanitize_float32
except Exception:  # pragma: no cover
    from pcm_utils import safe_resample, sanitize_float32


class RNNoiseFilter:
    """Fail-open RNNoise denoiser.

    The native RNNoise API processes 480-sample frames at 48 kHz. This wrapper
    accepts project-standard float32 mono audio, resamples when needed, and
    returns the original audio if anything goes wrong.
    """

    FRAME_SIZE = 480

    def __init__(self, sample_rate: int = 16000, enabled: Optional[bool] = None, dll_path: Optional[str] = None):
        self.sample_rate = int(sample_rate or 16000)
        env_enabled = os.environ.get("DITING_RNNOISE_ENABLED", os.environ.get("RNNOISE_ENABLED", "auto")).lower()
        if enabled is None:
            enabled = env_enabled not in {"0", "false", "off", "no"}
        self.enabled = bool(enabled)
        self.dll_path = dll_path or os.environ.get("RNNOISE_DLL_PATH") or ""
        self.error = ""
        self._dll = None
        self._state = None
        self._disabled_after_error = False
        self._load()

    @property
    def available(self) -> bool:
        return bool(self.enabled and self._dll is not None and self._state is not None and not self._disabled_after_error)

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "dll_path": self.dll_path,
            "error": self.error,
        }

    def _candidate_paths(self) -> list[Path]:
        here = Path(__file__).resolve()
        backend_dir = here.parents[1]
        paths: list[Path] = []
        if self.dll_path:
            paths.append(Path(self.dll_path))
        paths.extend([
            backend_dir / "third_party" / "rnnoise" / "rnnoise.dll",
            backend_dir / "rnnoise.dll",
        ])
        return paths

    def _load(self) -> None:
        if not self.enabled:
            self.error = "disabled"
            return
        load_errors = []
        dll = None
        chosen = ""
        for path in self._candidate_paths():
            try:
                if path.exists():
                    dll = ctypes.CDLL(str(path))
                    chosen = str(path)
                    break
            except Exception as exc:
                load_errors.append(f"{path}: {exc}")
        if dll is None:
            try:
                dll = ctypes.CDLL("rnnoise.dll")
                chosen = "PATH:rnnoise.dll"
            except Exception as exc:
                load_errors.append(f"PATH:rnnoise.dll: {exc}")

        if dll is None:
            self.error = "; ".join(load_errors) or "rnnoise.dll not found"
            return

        try:
            dll.rnnoise_create.argtypes = [ctypes.c_void_p]
            dll.rnnoise_create.restype = ctypes.c_void_p
            dll.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            dll.rnnoise_destroy.restype = None
            dll.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            dll.rnnoise_process_frame.restype = ctypes.c_float
            self._state = dll.rnnoise_create(None)
            if not self._state:
                self.error = "rnnoise_create returned null"
                return
            self._dll = dll
            self.dll_path = chosen
            self.error = ""
        except Exception as exc:
            self.error = f"RNNoise API init failed: {exc}"
            self._dll = None
            self._state = None

    def close(self) -> None:
        if self._dll is not None and self._state is not None:
            try:
                self._dll.rnnoise_destroy(self._state)
            except Exception:
                pass
        self._state = None

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def process_array(self, audio, sr: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Denoise a complete array; returns (audio, report)."""
        original = sanitize_float32(audio)
        report = self.status() | {"applied": False, "reason": "bypassed"}
        if original.size == 0:
            report["reason"] = "empty_audio"
            return original, report
        if not self.available:
            report["reason"] = "unavailable"
            return original, report

        try:
            src_sr = int(sr or self.sample_rate)
            work = safe_resample(original, src_sr, 48000)
            if work.size < self.FRAME_SIZE:
                report["reason"] = "shorter_than_one_frame"
                return original, report

            pad = (-work.size) % self.FRAME_SIZE
            if pad:
                work_padded = np.pad(work, (0, pad), mode="constant")
            else:
                work_padded = work

            out = np.empty_like(work_padded, dtype=np.float32)
            in_buf = (ctypes.c_float * self.FRAME_SIZE)()
            out_buf = (ctypes.c_float * self.FRAME_SIZE)()
            for start in range(0, work_padded.size, self.FRAME_SIZE):
                frame = np.asarray(work_padded[start:start + self.FRAME_SIZE], dtype=np.float32)
                for i, val in enumerate(frame):
                    in_buf[i] = float(val)
                self._dll.rnnoise_process_frame(self._state, out_buf, in_buf)
                out[start:start + self.FRAME_SIZE] = np.frombuffer(out_buf, dtype=np.float32, count=self.FRAME_SIZE)

            if pad:
                out = out[:-pad]
            denoised = safe_resample(out, 48000, src_sr)
            if denoised.size != original.size:
                if denoised.size > original.size:
                    denoised = denoised[:original.size]
                else:
                    denoised = np.pad(denoised, (0, original.size - denoised.size), mode="constant")
            denoised = sanitize_float32(denoised)
            if denoised.size == 0 or not np.isfinite(denoised).all() or float(np.max(np.abs(denoised))) < 1e-8:
                report["reason"] = "invalid_output"
                return original, report
            report.update({"applied": True, "reason": "ok"})
            return denoised, report
        except Exception as exc:
            self._disabled_after_error = True
            self.error = str(exc)
            report.update({"available": False, "error": self.error, "reason": "processing_error"})
            return original, report

    def process_chunk(self, audio, sr: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Realtime entrypoint. First version uses complete-chunk processing."""
        return self.process_array(audio, sr or self.sample_rate)
