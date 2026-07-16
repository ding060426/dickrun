"""Local Silero VAD segmentation for uploaded and canonical recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


class SileroFileVad:
    """Detect speech regions with the deployed sherpa-onnx Silero model."""

    provider_name = "sherpa-silero-file-vad"
    WINDOW_SIZE = 512

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.2,
        pre_padding_ms: int = 250,
        post_padding_ms: int = 450,
        num_threads: int = 1,
        detector_factory: Callable | None = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {self.model_path}")
        self.threshold = float(threshold)
        self.min_silence_duration = float(min_silence_duration)
        self.min_speech_duration = float(min_speech_duration)
        self.pre_padding_ms = max(0, int(pre_padding_ms))
        self.post_padding_ms = max(0, int(post_padding_ms))
        self.num_threads = max(1, int(num_threads))
        self._detector_factory = detector_factory

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        if sample_rate != 16000:
            raise ValueError(f"Silero file VAD requires 16000 Hz audio, got {sample_rate}")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(samples) == 0:
            return []

        detector = self._create_detector(sample_rate)
        raw_regions: list[tuple[int, int]] = []
        for offset in range(0, len(samples), self.WINDOW_SIZE):
            detector.accept_waveform(samples[offset:offset + self.WINDOW_SIZE])
            self._drain(detector, raw_regions)
        detector.flush()
        self._drain(detector, raw_regions)

        pre_padding = round(sample_rate * self.pre_padding_ms / 1000)
        post_padding = round(sample_rate * self.post_padding_ms / 1000)
        padded = [
            (max(0, start - pre_padding), min(len(samples), end + post_padding))
            for start, end in raw_regions
            if end > start
        ]
        merged: list[tuple[int, int]] = []
        for start, end in padded:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return [(start / sample_rate, end / sample_rate) for start, end in merged]

    def _create_detector(self, sample_rate: int):
        if self._detector_factory is not None:
            return self._detector_factory(None)
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(self.model_path),
                threshold=self.threshold,
                min_silence_duration=self.min_silence_duration,
                min_speech_duration=self.min_speech_duration,
                window_size=self.WINDOW_SIZE,
                max_speech_duration=60,
            ),
            sample_rate=sample_rate,
            num_threads=self.num_threads,
            provider="cpu",
        )
        return sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=120,
        )

    @staticmethod
    def _drain(detector, regions: list[tuple[int, int]]) -> None:
        while not detector.empty():
            segment = detector.front
            start = int(segment.start)
            regions.append((start, start + len(segment.samples)))
            detector.pop()
