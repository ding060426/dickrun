"""Temporal smoothing that preserves real speaker changes and overlaps."""

from __future__ import annotations

from .contracts import DiarizationSegment


def smooth_timeline(
    segments: list[DiarizationSegment],
    *,
    merge_gap_sec: float = 0.5,
) -> list[DiarizationSegment]:
    """Merge nearby same-speaker turns unless another speaker occupies the gap."""

    ordered = sorted(
        (segment for segment in segments if segment.end_sec > segment.start_sec),
        key=lambda item: (item.start_sec, item.end_sec, item.speaker_id),
    )
    merged: list[DiarizationSegment] = []
    for segment in ordered:
        match_index = next(
            (
                index
                for index in range(len(merged) - 1, -1, -1)
                if merged[index].speaker_id == segment.speaker_id
            ),
            None,
        )
        if match_index is None:
            merged.append(segment)
            continue

        previous = merged[match_index]
        gap = segment.start_sec - previous.end_sec
        gap_is_occupied = any(
            other.speaker_id != segment.speaker_id
            and other.start_sec < segment.start_sec
            and previous.end_sec < other.end_sec
            for other in merged[match_index + 1 :]
        )
        if gap <= merge_gap_sec and not gap_is_occupied:
            merged[match_index] = DiarizationSegment(
                start_sec=previous.start_sec,
                end_sec=max(previous.end_sec, segment.end_sec),
                speaker_id=segment.speaker_id,
                confidence=min(previous.confidence, segment.confidence),
                embedding_quality=min(
                    previous.embedding_quality,
                    segment.embedding_quality,
                ),
            )
        else:
            merged.append(segment)

    return _mark_overlaps(sorted(merged, key=lambda item: (item.start_sec, item.end_sec)))


def _mark_overlaps(segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
    marked: list[DiarizationSegment] = []
    for segment in segments:
        speakers = {
            other.speaker_id
            for other in segments
            if other.speaker_id != segment.speaker_id
            and segment.start_sec < other.end_sec
            and other.start_sec < segment.end_sec
        }
        marked.append(
            DiarizationSegment(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                speaker_id=segment.speaker_id,
                confidence=segment.confidence,
                overlap=bool(speakers),
                overlap_speakers=tuple(sorted(speakers | {segment.speaker_id}))
                if speakers
                else (),
                embedding_quality=segment.embedding_quality,
            )
        )
    return marked
