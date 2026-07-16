"""Local pyannote segmentation + 3D-Speaker diarization through sherpa-onnx."""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Callable

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

    def spawn(self, *, num_threads: int | None = None) -> "SherpaDiarizationBackend":
        """Create an independent worker; sherpa diarizers cannot share stream state."""

        return SherpaDiarizationBackend(
            self.segmentation_model,
            self.embedding_model,
            threshold=self.threshold,
            num_threads=num_threads or self.num_threads,
            min_turn_sec=self.min_turn_sec,
        )

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
        on_progress: Callable[[str, float], None] | None = None,
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
        if on_progress:
            on_progress("diarization", 0.0)
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
        if on_progress:
            on_progress("diarization", 1.0)
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


class SherpaSpeakerEmbedder:
    """Extract normalized 3D-Speaker embeddings for cross-chunk stitching."""

    def __init__(
        self,
        model: str | Path,
        *,
        num_threads: int = 1,
    ) -> None:
        self.model = Path(model)
        self.num_threads = max(1, int(num_threads))
        self._lock = threading.Lock()
        self._extractor = None

    def availability(self) -> tuple[bool, str]:
        if importlib.util.find_spec("sherpa_onnx") is None:
            return False, "sherpa-onnx is not installed"
        if not self.model.is_file():
            return False, f"missing speaker embedding model: {self.model}"
        return True, f"threads={self.num_threads}"

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason)
        samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if not len(samples):
            raise ValueError("speaker embedding requires non-empty audio")

        with self._lock:
            extractor = self._get_extractor()
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()
            if not extractor.is_ready(stream):
                raise ValueError(
                    f"speaker embedding needs more audio than {len(samples) / sample_rate:.2f}s"
                )
            embedding = np.asarray(extractor.compute(stream), dtype=np.float32)

        norm = float(np.linalg.norm(embedding))
        if not len(embedding) or not np.all(np.isfinite(embedding)) or norm <= 1e-8:
            raise RuntimeError("speaker embedding model returned an invalid vector")
        return embedding / norm

    def _get_extractor(self):
        if self._extractor is None:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model),
                num_threads=self.num_threads,
                provider="cpu",
            )
            if not config.validate():
                raise RuntimeError("invalid speaker embedding configuration")
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return self._extractor
