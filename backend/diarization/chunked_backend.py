"""Long-audio speaker diarization behind the existing backend interface."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contracts import (
    DiarizationBackend,
    DiarizationBackendResult,
    DiarizationSegment,
)
from .smoothing import smooth_timeline


logger = logging.getLogger("diarization")


@dataclass(frozen=True)
class ChunkedDiarizationConfig:
    """Runtime policy for switching from whole-file to chunked diarization."""

    enabled: bool = True
    long_audio_threshold_sec: float = 600.0
    target_chunk_sec: float = 300.0
    max_chunk_sec: float = 480.0
    overlap_sec: float = 2.0
    silence_search_sec: float = 15.0
    skip_silence_sec: float = 20.0
    max_workers: int = 2
    worker_threads: int = 2
    stitch_threshold: float = 0.75
    min_embedding_sec: float = 1.5
    max_embedding_sec: float = 10.0
    min_output_segment_sec: float = 0.1

    def __post_init__(self) -> None:
        threshold = max(0.0, float(self.long_audio_threshold_sec))
        target = max(1.0, float(self.target_chunk_sec))
        maximum = max(target, float(self.max_chunk_sec))
        object.__setattr__(self, "long_audio_threshold_sec", threshold)
        object.__setattr__(self, "target_chunk_sec", target)
        object.__setattr__(self, "max_chunk_sec", maximum)
        object.__setattr__(
            self,
            "overlap_sec",
            min(max(0.0, float(self.overlap_sec)), target / 4),
        )
        object.__setattr__(
            self,
            "silence_search_sec",
            max(0.0, float(self.silence_search_sec)),
        )
        object.__setattr__(
            self,
            "skip_silence_sec",
            max(0.0, float(self.skip_silence_sec)),
        )
        object.__setattr__(self, "max_workers", min(4, max(1, int(self.max_workers))))
        object.__setattr__(self, "worker_threads", min(8, max(1, int(self.worker_threads))))
        object.__setattr__(
            self,
            "stitch_threshold",
            min(1.0, max(0.0, float(self.stitch_threshold))),
        )
        minimum_embedding = max(0.5, float(self.min_embedding_sec))
        object.__setattr__(self, "min_embedding_sec", minimum_embedding)
        object.__setattr__(
            self,
            "max_embedding_sec",
            max(minimum_embedding, float(self.max_embedding_sec)),
        )
        object.__setattr__(
            self,
            "min_output_segment_sec",
            max(0.0, float(self.min_output_segment_sec)),
        )


@dataclass(frozen=True)
class _AudioChunk:
    index: int
    core_start_sec: float
    core_end_sec: float
    audio_start_sec: float
    audio_end_sec: float


@dataclass(frozen=True)
class _ChunkResult:
    chunk: _AudioChunk
    timeline: list[DiarizationSegment]


class ChunkedDiarizationBackend:
    """Add long-audio behavior without widening the meeting pipeline interface."""

    def __init__(
        self,
        whole_file_backend: DiarizationBackend,
        *,
        worker_factory: Callable[[], DiarizationBackend] | None = None,
        speech_detector=None,
        speaker_embedder=None,
        config: ChunkedDiarizationConfig | None = None,
    ) -> None:
        self.whole_file_backend = whole_file_backend
        self.worker_factory = worker_factory
        self.speech_detector = speech_detector
        self.speaker_embedder = speaker_embedder
        self.config = config or ChunkedDiarizationConfig()
        self.provider_name = f"chunked-{whole_file_backend.provider_name}"
        self._worker_state = threading.local()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(self.config.max_workers)),
            thread_name_prefix="huiwu-diarization",
        )

    def availability(self) -> tuple[bool, str]:
        available, detail = self.whole_file_backend.availability()
        if not available:
            return available, detail
        if not self.config.enabled:
            return True, f"{detail}; long-audio chunking=disabled"
        if (
            self.worker_factory is None
            or self.speech_detector is None
            or self.speaker_embedder is None
        ):
            return True, f"{detail}; long-audio chunking=fallback-only"
        embedder_status = getattr(self.speaker_embedder, "availability", None)
        if callable(embedder_status):
            embedding_available, embedding_detail = embedder_status()
            if not embedding_available:
                return True, (
                    f"{detail}; long-audio chunking=fallback-only "
                    f"({embedding_detail})"
                )
        return True, (
            f"{detail}; long-audio chunking>={self.config.long_audio_threshold_sec:g}s, "
            f"workers={max(1, self.config.max_workers)}"
        )

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> DiarizationBackendResult:
        started_at = time.monotonic()
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
        if not self.config.enabled or duration < self.config.long_audio_threshold_sec:
            if on_progress:
                on_progress("diarization", 0.0)
            outcome = self._coerce_result(
                self.whole_file_backend.diarize(
                    audio,
                    sample_rate,
                    num_speakers=num_speakers,
                )
            )
            outcome.metadata.update(
                {
                    "chunked": False,
                    "chunk_count": 1 if duration > 0 else 0,
                    "worker_count": 1 if duration > 0 else 0,
                    "skipped_silence_sec": 0.0,
                    "processed_audio_sec": round(duration, 3),
                    "diarization_elapsed_sec": round(time.monotonic() - started_at, 3),
                }
            )
            if on_progress:
                on_progress("diarization", 1.0)
            return outcome
        if (
            self.worker_factory is None
            or self.speech_detector is None
            or self.speaker_embedder is None
        ):
            outcome = self._coerce_result(
                self.whole_file_backend.diarize(
                    audio,
                    sample_rate,
                    num_speakers=num_speakers,
                )
            )
            outcome.metadata.update(
                {
                    "chunked": False,
                    "chunk_fallback": "chunk dependencies unavailable",
                    "chunk_count": 1,
                    "worker_count": 1,
                    "skipped_silence_sec": 0.0,
                    "processed_audio_sec": round(duration, 3),
                    "diarization_elapsed_sec": round(time.monotonic() - started_at, 3),
                }
            )
            return outcome

        if on_progress:
            on_progress("diarization planning", 0.0)
        try:
            regions = self.speech_detector.detect(audio, sample_rate)
            chunks = self._plan_chunks(regions, duration)
        except Exception as error:
            return self._whole_file_fallback(
                audio,
                sample_rate,
                num_speakers,
                error,
                started_at,
                on_progress,
            )
        if not chunks:
            if on_progress:
                on_progress("diarization 0/0", 1.0)
            return DiarizationBackendResult(
                timeline=[],
                metadata={
                    "chunked": True,
                    "chunk_count": 0,
                    "worker_count": 0,
                    "skipped_silence_sec": round(duration, 3),
                    "processed_audio_sec": 0.0,
                    "diarization_elapsed_sec": round(time.monotonic() - started_at, 3),
                },
            )

        future_to_chunk = {
            self._executor.submit(
                self._process_chunk,
                chunk,
                audio,
                sample_rate,
                num_speakers,
            ): chunk
            for chunk in chunks
        }
        chunk_results: list[_ChunkResult] = []
        try:
            for completed, future in enumerate(
                concurrent.futures.as_completed(future_to_chunk),
                1,
            ):
                chunk_results.append(future.result())
                if on_progress:
                    on_progress(
                        f"diarization {completed}/{len(chunks)}",
                        completed / len(chunks),
                    )
        except Exception as error:
            for future in future_to_chunk:
                future.cancel()
            concurrent.futures.wait(future_to_chunk)
            return self._whole_file_fallback(
                audio,
                sample_rate,
                num_speakers,
                error,
                started_at,
                on_progress,
            )
        chunk_results.sort(key=lambda result: result.chunk.index)
        try:
            stitched = self._stitch_speakers(
                chunk_results,
                audio,
                sample_rate,
                num_speakers=num_speakers,
            )
        except Exception as error:
            return self._whole_file_fallback(
                audio,
                sample_rate,
                num_speakers,
                error,
                started_at,
                on_progress,
            )
        core_audio_sec = sum(
            chunk.core_end_sec - chunk.core_start_sec
            for chunk in chunks
        )
        processed_audio_sec = sum(
            chunk.audio_end_sec - chunk.audio_start_sec
            for chunk in chunks
        )
        return DiarizationBackendResult(
            timeline=smooth_timeline(stitched),
            metadata={
                "chunked": True,
                "chunk_count": len(chunks),
                "worker_count": min(max(1, self.config.max_workers), len(chunks)),
                "skipped_silence_sec": round(max(0.0, duration - core_audio_sec), 3),
                "processed_audio_sec": round(processed_audio_sec, 3),
                "diarization_elapsed_sec": round(time.monotonic() - started_at, 3),
            },
        )

    def _whole_file_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None,
        error: Exception,
        started_at: float,
        on_progress: Callable[[str, float], None] | None,
    ) -> DiarizationBackendResult:
        logger.warning(
            "Chunked diarization failed; retrying the whole file: %s",
            error,
        )
        if on_progress:
            on_progress("diarization whole-file fallback", 0.0)
        outcome = self._coerce_result(
            self.whole_file_backend.diarize(
                audio,
                sample_rate,
                num_speakers=num_speakers,
            )
        )
        outcome.metadata.update(
            {
                "chunked": False,
                "chunk_fallback": str(error),
                "chunk_count": 1,
                "worker_count": 1,
                "skipped_silence_sec": 0.0,
                "processed_audio_sec": round(len(audio) / sample_rate, 3),
                "diarization_elapsed_sec": round(time.monotonic() - started_at, 3),
            }
        )
        if on_progress:
            on_progress("diarization whole-file fallback", 1.0)
        return outcome

    def _process_chunk(
        self,
        chunk: _AudioChunk,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None,
    ) -> _ChunkResult:
        worker = getattr(self._worker_state, "backend", None)
        if worker is None:
            worker = self.worker_factory()
            self._worker_state.backend = worker
        start_sample = round(chunk.audio_start_sec * sample_rate)
        end_sample = round(chunk.audio_end_sec * sample_rate)
        chunk_audio = audio[start_sample:end_sample]
        try:
            local_value = worker.diarize(
                chunk_audio,
                sample_rate,
                # The user-supplied count describes the whole meeting. A chunk can
                # contain only a subset, so constrain only the global stitch step.
                num_speakers=None,
            )
        except Exception:
            logger.warning(
                "Diarization chunk %d failed; retrying with a fresh worker",
                chunk.index + 1,
            )
            worker = self.worker_factory()
            self._worker_state.backend = worker
            local_value = worker.diarize(
                chunk_audio,
                sample_rate,
                num_speakers=None,
            )
        local_outcome = self._coerce_result(local_value)
        return _ChunkResult(
            chunk=chunk,
            timeline=self._restore_global_time(local_outcome.timeline, chunk),
        )

    @staticmethod
    def _coerce_result(
        value: DiarizationBackendResult | list[DiarizationSegment],
    ) -> DiarizationBackendResult:
        if isinstance(value, DiarizationBackendResult):
            return value
        return DiarizationBackendResult(timeline=list(value))

    def _plan_chunks(
        self,
        raw_regions,
        duration_sec: float,
    ) -> list[_AudioChunk]:
        regions = sorted(
            (
                (max(0.0, float(region[0])), min(duration_sec, float(region[1])))
                for region in raw_regions
            ),
            key=lambda item: (item[0], item[1]),
        )
        regions = [region for region in regions if region[1] > region[0]]
        if not regions:
            return []

        spans: list[tuple[float, float, list[tuple[float, float]]]] = []
        span_regions = [regions[0]]
        for region in regions[1:]:
            if region[0] - span_regions[-1][1] >= self.config.skip_silence_sec:
                spans.append((span_regions[0][0], span_regions[-1][1], span_regions))
                span_regions = [region]
            else:
                span_regions.append(region)
        spans.append((span_regions[0][0], span_regions[-1][1], span_regions))

        cores: list[tuple[float, float]] = []
        for span_start, span_end, items in spans:
            cores.extend(self._split_span(span_start, span_end, items))

        overlap = max(0.0, self.config.overlap_sec)
        return [
            _AudioChunk(
                index=index,
                core_start_sec=start,
                core_end_sec=end,
                audio_start_sec=max(0.0, start - overlap),
                audio_end_sec=min(duration_sec, end + overlap),
            )
            for index, (start, end) in enumerate(cores)
        ]

    def _split_span(
        self,
        span_start: float,
        span_end: float,
        regions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        max_length = max(1.0, self.config.max_chunk_sec)
        target = min(max_length, max(1.0, self.config.target_chunk_sec))
        search = max(0.0, self.config.silence_search_sec)
        silence_midpoints = [
            (left[1] + right[0]) / 2
            for left, right in zip(regions, regions[1:])
            if right[0] > left[1]
        ]
        chunks: list[tuple[float, float]] = []
        cursor = span_start
        while span_end - cursor > max_length:
            preferred = cursor + target
            latest = cursor + max_length
            candidates = [
                point
                for point in silence_midpoints
                if cursor + 0.5 < point <= latest
                and abs(point - preferred) <= search
            ]
            boundary = (
                min(candidates, key=lambda point: abs(point - preferred))
                if candidates
                else min(preferred, latest)
            )
            chunks.append((cursor, boundary))
            cursor = boundary
        chunks.append((cursor, span_end))
        return chunks

    def _restore_global_time(
        self,
        timeline: list[DiarizationSegment],
        chunk: _AudioChunk,
    ) -> list[DiarizationSegment]:
        restored: list[DiarizationSegment] = []
        for segment in timeline:
            start = max(
                chunk.core_start_sec,
                chunk.audio_start_sec + segment.start_sec,
            )
            end = min(
                chunk.core_end_sec,
                chunk.audio_start_sec + segment.end_sec,
            )
            if end - start < self.config.min_output_segment_sec:
                continue
            restored.append(
                DiarizationSegment(
                    start_sec=start,
                    end_sec=end,
                    speaker_id=self._local_key(
                        chunk.index,
                        segment.speaker_id,
                    ),
                    confidence=segment.confidence,
                    overlap=segment.overlap,
                    overlap_speakers=tuple(
                        self._local_key(chunk.index, speaker)
                        for speaker in segment.overlap_speakers
                    ),
                    embedding_quality=segment.embedding_quality,
                )
            )
        return restored

    def _stitch_speakers(
        self,
        chunk_results: list[_ChunkResult],
        audio: np.ndarray,
        sample_rate: int,
        *,
        num_speakers: int | None,
    ) -> list[DiarizationSegment]:
        segments = [
            segment
            for result in chunk_results
            for segment in result.timeline
        ]
        if not segments:
            return []

        first_seen: dict[str, float] = {}
        for segment in segments:
            first_seen[segment.speaker_id] = min(
                first_seen.get(segment.speaker_id, segment.start_sec),
                segment.start_sec,
            )
        local_keys = sorted(first_seen, key=lambda key: (first_seen[key], key))

        if self.speaker_embedder is None:
            suffixes = {key: key.split(":", 1)[-1] for key in local_keys}
            ordered_suffixes = []
            for key in local_keys:
                suffix = suffixes[key]
                if suffix not in ordered_suffixes:
                    ordered_suffixes.append(suffix)
            mapping = {
                key: f"SPEAKER_{ordered_suffixes.index(suffixes[key]):02d}"
                for key in local_keys
            }
        else:
            prototypes = {
                key: self._speaker_prototype(key, segments, audio, sample_rate)
                for key in local_keys
            }
            mapping = self._cluster_prototypes(
                local_keys,
                prototypes,
                num_speakers=num_speakers,
            )

        remapped: list[DiarizationSegment] = []
        for segment in segments:
            speaker_id = mapping[segment.speaker_id]
            overlap_speakers = tuple(
                sorted(
                    {
                        mapping.get(speaker, speaker_id)
                        for speaker in segment.overlap_speakers
                    }
                )
            )
            remapped.append(
                DiarizationSegment(
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    speaker_id=speaker_id,
                    confidence=segment.confidence,
                    overlap=segment.overlap,
                    overlap_speakers=overlap_speakers,
                    embedding_quality=segment.embedding_quality,
                )
            )
        return remapped

    def _speaker_prototype(
        self,
        local_key: str,
        segments: list[DiarizationSegment],
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray | None:
        candidates = sorted(
            (
                segment
                for segment in segments
                if segment.speaker_id == local_key
                and not segment.overlap
                and segment.duration >= self.config.min_embedding_sec
            ),
            key=lambda segment: (-segment.duration, segment.start_sec),
        )
        remaining = max(self.config.min_embedding_sec, self.config.max_embedding_sec)
        samples: list[np.ndarray] = []
        for segment in candidates:
            if remaining <= 0:
                break
            duration = min(segment.duration, remaining)
            start = round(segment.start_sec * sample_rate)
            end = min(len(audio), start + round(duration * sample_rate))
            if end > start:
                samples.append(audio[start:end])
                remaining -= (end - start) / sample_rate
        if not samples:
            return None
        embedding = np.asarray(
            self.speaker_embedder.extract(np.concatenate(samples), sample_rate),
            dtype=np.float32,
        ).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not len(embedding) or not np.all(np.isfinite(embedding)) or norm <= 1e-8:
            return None
        return embedding / norm

    def _cluster_prototypes(
        self,
        local_keys: list[str],
        prototypes: dict[str, np.ndarray | None],
        *,
        num_speakers: int | None,
    ) -> dict[str, str]:
        valid_keys = [key for key in local_keys if prototypes[key] is not None]
        if num_speakers is not None and len(valid_keys) != len(local_keys):
            raise RuntimeError(
                "known speaker count requires a usable voiceprint for every chunk speaker"
            )
        if num_speakers is not None and int(num_speakers) > len(valid_keys):
            raise RuntimeError(
                "known speaker count exceeds the number of chunk speaker voiceprints"
            )
        cluster_by_key: dict[str, int] = {}
        if len(valid_keys) == 1:
            cluster_by_key[valid_keys[0]] = 1
        elif len(valid_keys) > 1:
            from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
            from scipy.spatial.distance import pdist

            matrix = np.stack([prototypes[key] for key in valid_keys])
            distances = pdist(matrix, metric="cosine")
            tree = linkage(distances, method="average")
            if num_speakers is not None:
                labels = cut_tree(
                    tree,
                    n_clusters=[max(1, int(num_speakers))],
                ).reshape(-1) + 1
            else:
                labels = fcluster(
                    tree,
                    t=max(0.0, 1.0 - self.config.stitch_threshold),
                    criterion="distance",
                )
            cluster_by_key.update(
                {key: int(label) for key, label in zip(valid_keys, labels)}
            )

        next_cluster = max(cluster_by_key.values(), default=0) + 1
        for key in local_keys:
            if key not in cluster_by_key:
                cluster_by_key[key] = next_cluster
                next_cluster += 1

        ordered_clusters: list[int] = []
        for key in local_keys:
            cluster = cluster_by_key[key]
            if cluster not in ordered_clusters:
                ordered_clusters.append(cluster)
        return {
            key: f"SPEAKER_{ordered_clusters.index(cluster_by_key[key]):02d}"
            for key in local_keys
        }

    @staticmethod
    def _local_key(chunk_index: int, speaker_id: str) -> str:
        return f"{chunk_index}:{speaker_id}"
