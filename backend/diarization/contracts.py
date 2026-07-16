"""Dependency-light contracts for interchangeable diarization backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class SpeechRegion:
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass(frozen=True)
class DiarizationSegment:
    start_sec: float
    end_sec: float
    speaker_id: str
    confidence: float = 0.8
    overlap: bool = False
    overlap_speakers: tuple[str, ...] = field(default_factory=tuple)
    embedding_quality: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class DiarizationBackendResult:
    timeline: list[DiarizationSegment]
    metadata: dict[str, object] = field(default_factory=dict)


class DiarizationBackend(Protocol):
    provider_name: str

    def availability(self) -> tuple[bool, str]: ...

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> DiarizationBackendResult | list[DiarizationSegment]: ...
