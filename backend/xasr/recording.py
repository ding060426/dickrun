"""Durable live PCM recording with atomic completion and recovery files."""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordingResult:
    path: Path
    part_path: Path
    received_samples: int
    duration_ms: int


class LiveRecording:
    """Persist one 16 kHz mono PCM stream behind a two-method interface."""

    def __init__(
        self,
        directory: str | Path,
        recording_id: str,
        *,
        sample_rate: int = 16000,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", recording_id).strip("._")
        if not safe_id:
            raise ValueError("recording_id must contain a safe character")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.path = self.directory / f"{safe_id}.wav"
        self.part_path = self.directory / f"{safe_id}.wav.part"
        if self.path.exists() or self.part_path.exists():
            raise FileExistsError(f"recording already exists: {safe_id}")
        self.received_samples = 0
        self._closed = False
        self._result: RecordingResult | None = None
        self._wave = wave.open(str(self.part_path), "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(sample_rate)

    def append_pcm_s16le(self, payload: bytes) -> None:
        if self._closed:
            raise RuntimeError("recording is already closed")
        if not payload or len(payload) % 2:
            raise ValueError("PCM payload must contain complete int16 samples")
        self._wave.writeframesraw(payload)
        self.received_samples += len(payload) // 2

    def finalize(self) -> RecordingResult:
        if self._result is not None:
            return self._result
        if not self._closed:
            self._wave.close()
            self._closed = True
        self.part_path.replace(self.path)
        self._result = RecordingResult(
            path=self.path,
            part_path=self.part_path,
            received_samples=self.received_samples,
            duration_ms=round(self.received_samples * 1000 / self.sample_rate),
        )
        return self._result

    def abort(self) -> Path:
        if self._result is not None:
            return self._result.path
        if not self._closed:
            self._wave.close()
            self._closed = True
        return self.part_path
