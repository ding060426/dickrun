"""Local pyannote segmentation + 3D-Speaker diarization through sherpa-onnx."""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import numpy as np

from .contracts import DiarizationSegment
from .smoothing import smooth_timeline


class SherpaDiarizationBackend:
    provider_name = "sherpa-pyannote-3dspeaker"

    def __init__(
        self,
        segmentation_model: str | Path,
        embedding_model: str | Path,
        *,
        threshold: float = 0.8,
        num_threads: int | None = None,
        min_turn_sec: float = 0.5,
    ) -> None:
        self.segmentation_model = Path(segmentation_model)
        self.embedding_model = Path(embedding_model)
        self.threshold = float(threshold)
        self.num_threads = max(
            1,
            int(num_threads or min(8, max(2, (os.cpu_count() or 4) // 2))),
        )
        self.min_turn_sec = max(0.0, float(min_turn_sec))
        self._lock = threading.Lock()
        self._diarizer_key: int | None = None
        self._diarizer = None

    def availability(self) -> tuple[bool, str]:
        if importlib.util.find_spec("sherpa_onnx") is None:
            return False, "sherpa-onnx is not installed"
        missing = [
            str(path)
            for path in (self.segmentation_model, self.embedding_model)
            if not path.is_file()
        ]
        if missing:
            return False, f"missing diarization model(s): {', '.join(missing)}"
        return True, f"threshold={self.threshold:g}, threads={self.num_threads}"

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason)
        if sample_rate != 16_000:
            raise ValueError(f"diarization requires 16000 Hz audio, got {sample_rate}")
        if num_speakers is not None and not 1 <= int(num_speakers) <= 20:
            raise ValueError("num_speakers must be between 1 and 20")

        cluster_count = int(num_speakers) if num_speakers is not None else -1
        samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._diarizer is None or self._diarizer_key != cluster_count:
                self._diarizer = self._create_diarizer(cluster_count)
                self._diarizer_key = cluster_count
            raw_result = self._diarizer.process(samples)

        segments = [
            DiarizationSegment(
                start_sec=max(0.0, float(item.start)),
                end_sec=min(len(samples) / sample_rate, float(item.end)),
                speaker_id=f"SPEAKER_{int(item.speaker):02d}",
                confidence=0.8,
            )
            for item in raw_result.sort_by_start_time()
            if float(item.end) > float(item.start)
        ]
        return smooth_timeline(segments, merge_gap_sec=self.min_turn_sec)

    def _create_diarizer(self, cluster_count: int):
        import sherpa_onnx

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_model)
                ),
                num_threads=self.num_threads,
                provider="cpu",
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_model),
                num_threads=self.num_threads,
                provider="cpu",
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=cluster_count,
                threshold=self.threshold,
            ),
            min_duration_on=0.3,
            min_duration_off=self.min_turn_sec,
        )
        if not config.validate():
            raise RuntimeError("invalid sherpa-onnx diarization configuration")
        return sherpa_onnx.OfflineSpeakerDiarization(config)
