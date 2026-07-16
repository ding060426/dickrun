"""Small PCM helpers for realtime audio paths.

All helpers are fail-open: bad input returns an empty or clipped float32 array
instead of raising into the WebSocket loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - optional dependency fallback
    resample_poly = None


def sanitize_float32(audio) -> np.ndarray:
    """Return a safe mono float32 array clipped to [-1, 1]."""
    try:
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    if data.size == 0:
        return data.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def int16_pcm_to_float32(payload: bytes) -> np.ndarray:
    """Decode little-endian PCM S16LE bytes into float32 mono samples."""
    if not payload:
        return np.zeros(0, dtype=np.float32)
    usable = len(payload) - (len(payload) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    data = np.frombuffer(payload[:usable], dtype="<i2").astype(np.float32)
    return np.clip(data / 32768.0, -1.0, 1.0).astype(np.float32)


def float32_to_int16_pcm(audio) -> bytes:
    """Encode float32 samples into little-endian PCM S16LE bytes."""
    data = sanitize_float32(audio)
    if data.size == 0:
        return b""
    return (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def rms_level(audio) -> float:
    """Compute a stable RMS level in [0, 1]-ish range."""
    data = sanitize_float32(audio)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data * data) + 1e-12))


def safe_resample(audio, src_sr: Optional[int], dst_sr: int = 16000) -> np.ndarray:
    """Resample mono float32 audio; returns original data if resampling fails."""
    data = sanitize_float32(audio)
    if data.size == 0:
        return data
    try:
        src = int(src_sr or dst_sr)
        dst = int(dst_sr)
    except Exception:
        return data
    if src <= 0 or dst <= 0 or src == dst:
        return data

    try:
        if resample_poly is not None:
            from math import gcd
            g = gcd(src, dst)
            up = dst // g
            down = src // g
            return sanitize_float32(resample_poly(data, up, down))

        # Lightweight interpolation fallback. Good enough for fail-open paths.
        old_x = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
        new_len = max(1, int(round(data.size * dst / src)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return sanitize_float32(np.interp(new_x, old_x, data).astype(np.float32))
    except Exception:
        return data
