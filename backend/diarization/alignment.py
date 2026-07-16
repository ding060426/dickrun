"""Time-overlap alignment between the independent ASR and speaker timelines."""

from __future__ import annotations

from collections import defaultdict

from .contracts import DiarizationSegment


def overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def align_result(result, timeline: list[DiarizationSegment]):
    """Assign by total time overlap, never by the ASR segment midpoint."""

    durations: dict[str, float] = defaultdict(float)
    confidence_weight: dict[str, float] = defaultdict(float)
    overlap_speakers: set[str] = set()
    relevant: list[DiarizationSegment] = []
    for segment in timeline:
        duration = overlap_duration(
            result.start_sec,
            result.end_sec,
            segment.start_sec,
            segment.end_sec,
        )
        if duration <= 0:
            continue
        relevant.append(segment)
        durations[segment.speaker_id] += duration
        confidence_weight[segment.speaker_id] += duration * segment.confidence
        if segment.overlap:
            overlap_speakers.update(segment.overlap_speakers)

    if not durations:
        result.speaker_id = "UNKNOWN"
        result.speaker_confidence = 0.0
        result.overlap = False
        result.overlap_speakers = []
        return result

    ranked = sorted(durations, key=lambda speaker: (-durations[speaker], speaker))
    primary = ranked[0]
    total = sum(durations.values())
    model_confidence = confidence_weight[primary] / max(durations[primary], 1e-9)
    result.speaker_id = primary
    result.speaker_confidence = round(
        model_confidence * durations[primary] / max(total, 1e-9),
        3,
    )
    result.overlap = bool(overlap_speakers)
    result.overlap_speakers = sorted(overlap_speakers)

    for word in getattr(result, "words", []):
        word_durations = {
            segment.speaker_id: overlap_duration(
                word.start_sec,
                word.end_sec,
                segment.start_sec,
                segment.end_sec,
            )
            for segment in relevant
        }
        if word_durations and max(word_durations.values()) > 0:
            word.speaker_id = sorted(
                word_durations,
                key=lambda speaker: (-word_durations[speaker], speaker),
            )[0]
    return result


def align_results(results: list, timeline: list[DiarizationSegment]) -> list:
    return [align_result(result, timeline) for result in results]


def boundary_intervals(
    result,
    timeline: list[DiarizationSegment],
) -> list[DiarizationSegment]:
    """Return non-overlap speaker intervals inside one ASR segment."""

    relevant = [
        segment
        for segment in timeline
        if overlap_duration(
            result.start_sec,
            result.end_sec,
            segment.start_sec,
            segment.end_sec,
        )
        > 0
    ]
    if any(segment.overlap for segment in relevant):
        return []

    clipped: list[DiarizationSegment] = []
    for segment in relevant:
        start = max(result.start_sec, segment.start_sec)
        end = min(result.end_sec, segment.end_sec)
        if end <= start:
            continue
        current = DiarizationSegment(
            start_sec=start,
            end_sec=end,
            speaker_id=segment.speaker_id,
            confidence=segment.confidence,
            embedding_quality=segment.embedding_quality,
        )
        if (
            clipped
            and clipped[-1].speaker_id == current.speaker_id
            and current.start_sec <= clipped[-1].end_sec + 0.05
        ):
            previous = clipped[-1]
            clipped[-1] = DiarizationSegment(
                start_sec=previous.start_sec,
                end_sec=max(previous.end_sec, current.end_sec),
                speaker_id=current.speaker_id,
                confidence=min(previous.confidence, current.confidence),
                embedding_quality=min(
                    previous.embedding_quality,
                    current.embedding_quality,
                ),
            )
        else:
            clipped.append(current)

    if len({segment.speaker_id for segment in clipped}) < 2:
        return []
    return clipped


def deduplicate_boundary(previous_text: str, current_text: str) -> str:
    """Remove only a >=3-character suffix/prefix duplicate caused by padding."""

    previous = previous_text or ""
    current = current_text or ""
    limit = min(len(previous), len(current), 40)
    for size in range(limit, 2, -1):
        if previous[-size:] == current[:size]:
            return current[size:].lstrip()
    return current
