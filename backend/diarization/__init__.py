"""Offline speaker diarization and ASR alignment for DiTing."""

from .chunked_backend import ChunkedDiarizationBackend, ChunkedDiarizationConfig
from .contracts import DiarizationBackendResult, DiarizationSegment, SpeechRegion
from .pipeline import DiarizationRun, OfflineMeetingPipeline
from .sherpa_backend import SherpaDiarizationBackend, SherpaSpeakerEmbedder

__all__ = [
    "ChunkedDiarizationBackend",
    "ChunkedDiarizationConfig",
    "DiarizationBackendResult",
    "DiarizationRun",
    "DiarizationSegment",
    "OfflineMeetingPipeline",
    "SherpaDiarizationBackend",
    "SherpaSpeakerEmbedder",
    "SpeechRegion",
]
