"""Canonical audio loading shared by ASR, VAD, and speaker diarization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CANONICAL_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioBuffer:
    """A single 16 kHz mono float32 timeline used by every speech module."""

    samples: np.ndarray
    sample_rate: int = CANONICAL_SAMPLE_RATE
    source_path: str = ""

    def __post_init__(self) -> None:
        samples = np.ascontiguousarray(self.samples, dtype=np.float32).reshape(-1)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio contains NaN or infinite samples")
        object.__setattr__(self, "samples", np.clip(samples, -1.0, 1.0))

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def slice(self, start_sec: float, end_sec: float) -> np.ndarray:
        start = max(0, min(len(self.samples), round(start_sec * self.sample_rate)))
        end = max(start, min(len(self.samples), round(end_sec * self.sample_rate)))
        return self.samples[start:end]


def load_audio_buffer(
    file_path: str | Path,
    *,
    target_sample_rate: int = CANONICAL_SAMPLE_RATE,
) -> AudioBuffer:
    """Decode an audio file once and return the canonical shared timeline."""

    path = Path(file_path)
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        samples = data.mean(axis=1)
    except Exception as soundfile_error:
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(path)
            sample_rate = audio.frame_rate
            raw = np.asarray(audio.get_array_of_samples(), dtype=np.float32)
            raw = raw.reshape(-1, audio.channels).mean(axis=1)
            max_pcm = float(1 << (8 * audio.sample_width - 1))
            samples = raw / max_pcm
        except Exception as pydub_error:
            try:
                import librosa

                samples, sample_rate = librosa.load(
                    path,
                    sr=None,
                    mono=True,
                    dtype=np.float32,
                )
            except Exception as librosa_error:
                raise RuntimeError(
                    f"Cannot decode audio file {path}: "
                    f"soundfile={soundfile_error}; pydub={pydub_error}; "
                    f"librosa={librosa_error}"
                ) from librosa_error

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != target_sample_rate and len(samples):
        from scipy.signal import resample_poly

        divisor = math.gcd(int(sample_rate), int(target_sample_rate))
        samples = resample_poly(
            samples,
            target_sample_rate // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
        sample_rate = target_sample_rate
    elif sample_rate != target_sample_rate:
        sample_rate = target_sample_rate

    return AudioBuffer(
        samples=samples,
        sample_rate=int(sample_rate),
        source_path=str(path),
    )
