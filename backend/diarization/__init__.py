"""Offline speaker diarization and ASR alignment for DiTing."""

from .contracts import DiarizationSegment, SpeechRegion
from .pipeline import DiarizationRun, OfflineMeetingPipeline
from .sherpa_backend import SherpaDiarizationBackend

__all__ = [
    "DiarizationRun",
    "DiarizationSegment",
    "OfflineMeetingPipeline",
    "SherpaDiarizationBackend",
    "SpeechRegion",
]
