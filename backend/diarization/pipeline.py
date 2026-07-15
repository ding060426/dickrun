"""Dual-track offline meeting pipeline: diarization and ASR, then alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from audio_buffer import AudioBuffer, load_audio_buffer

from .alignment import (
    align_results,
    boundary_intervals,
    deduplicate_boundary,
)
from .contracts import DiarizationBackend, DiarizationSegment


logger = logging.getLogger("diarization")


@dataclass
class DiarizationRun:
    results: list
    timeline: list[DiarizationSegment] = field(default_factory=list)
    enabled: bool = False
    applied: bool = False
    provider: str = "disabled"
    reason: str = ""
    boundary_redecoded_segments: int = 0

    @property
    def speakers(self) -> list[dict]:
        durations: dict[str, float] = {}
        confidence: dict[str, list[float]] = {}
        for segment in self.timeline:
            durations[segment.speaker_id] = (
                durations.get(segment.speaker_id, 0.0) + segment.duration
            )
            confidence.setdefault(segment.speaker_id, []).append(segment.confidence)
        return [
            {
                "id": speaker_id,
                "name": None,
                "duration": round(durations[speaker_id], 3),
                "confidence": round(
                    sum(confidence[speaker_id]) / len(confidence[speaker_id]),
                    3,
                ),
            }
            for speaker_id in sorted(durations)
        ]

    def metadata(self) -> dict:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "provider": self.provider,
            "reason": self.reason or None,
            "speaker_count": len(self.speakers),
            "timeline_segments": len(self.timeline),
            "boundary_redecoded_segments": self.boundary_redecoded_segments,
        }


class OfflineMeetingPipeline:
    """Keep ASR and diarization independent until their timestamps are aligned."""

    def __init__(
        self,
        backend: DiarizationBackend,
        *,
        boundary_pre_padding_ms: int = 250,
        boundary_post_padding_ms: int = 400,
        min_redecode_sec: float = 0.35,
    ) -> None:
        self.backend = backend
        self.boundary_pre_padding_ms = max(0, int(boundary_pre_padding_ms))
        self.boundary_post_padding_ms = max(0, int(boundary_post_padding_ms))
        self.min_redecode_sec = max(0.1, float(min_redecode_sec))

    def status(self) -> dict:
        available, detail = self.backend.availability()
        return {
            "available": available,
            "provider": self.backend.provider_name,
            "detail": detail,
        }

    def process_file(
        self,
        file_path: str | Path,
        engine,
        *,
        enable_diarization: bool = True,
        num_speakers: int | None = None,
        on_segment: Callable | None = None,
        on_progress: Callable | None = None,
    ) -> DiarizationRun:
        audio = load_audio_buffer(file_path)
        provider = self.backend.provider_name
        if not enable_diarization:
            results = engine.process_file(
                str(file_path),
                on_segment=on_segment,
                on_progress=on_progress,
                audio_buffer=audio,
            )
            return DiarizationRun(
                results=results,
                enabled=False,
                applied=False,
                provider="disabled",
            )

        available, reason = self.backend.availability()
        if not available:
            logger.warning("Diarization unavailable; using ASR-only output: %s", reason)
            results = engine.process_file(
                str(file_path),
                on_segment=on_segment,
                on_progress=on_progress,
                audio_buffer=audio,
            )
            return DiarizationRun(
                results=results,
                enabled=True,
                applied=False,
                provider=provider,
                reason=reason,
            )

        if on_progress:
            on_progress("diarization", 0.05)
        timeline = self.backend.diarize(
            audio.samples,
            audio.sample_rate,
            num_speakers=num_speakers,
        )
        if on_progress:
            on_progress("asr", 0.35)

        def asr_progress(stage: str, fraction: float) -> None:
            if on_progress:
                on_progress(stage, 0.35 + 0.55 * fraction)

        base_results = engine.process_file(
            str(file_path),
            on_segment=None,
            on_progress=asr_progress,
            audio_buffer=audio,
        )
        results, boundary_redecoded = self._boundary_redecode(
            base_results,
            timeline,
            audio,
            engine,
        )
        align_results(results, timeline)
        self._deduplicate_adjacent(results)

        if on_progress:
            on_progress("alignment", 0.95)
        if on_segment:
            for index, result in enumerate(results, 1):
                on_segment(result, index, len(results))
        if on_progress:
            on_progress("done", 1.0)

        return DiarizationRun(
            results=results,
            timeline=timeline,
            enabled=True,
            applied=True,
            provider=provider,
            boundary_redecoded_segments=boundary_redecoded,
        )

    def _boundary_redecode(
        self,
        base_results: list,
        timeline: list[DiarizationSegment],
        audio: AudioBuffer,
        engine,
    ) -> tuple[list, int]:
        if not hasattr(engine, "recognize_interval"):
            return base_results, 0
        refined_results: list = []
        redecoded = 0
        for result in base_results:
            intervals = boundary_intervals(result, timeline)
            if not intervals or any(
                interval.duration < self.min_redecode_sec for interval in intervals
            ):
                refined_results.append(result)
                continue

            candidates = []
            for interval in intervals:
                candidate = engine.recognize_interval(
                    audio,
                    interval.start_sec,
                    interval.end_sec,
                    pre_padding_ms=self.boundary_pre_padding_ms,
                    post_padding_ms=self.boundary_post_padding_ms,
                )
                if candidate is None or not candidate.text.strip():
                    candidates = []
                    break
                candidate.speaker_id = interval.speaker_id
                candidate.speaker_confidence = interval.confidence
                candidates.append(candidate)
            if candidates:
                refined_results.extend(candidates)
                redecoded += len(candidates)
            else:
                refined_results.append(result)
        return refined_results, redecoded

    @staticmethod
    def _deduplicate_adjacent(results: list) -> None:
        for previous, current in zip(results, results[1:]):
            text = deduplicate_boundary(previous.text, current.text)
            raw_text = deduplicate_boundary(previous.raw_text, current.raw_text)
            if text.strip():
                current.text = text
            if raw_text.strip():
                current.raw_text = raw_text
