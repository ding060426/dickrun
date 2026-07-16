"""Thread-safe in-process meeting metadata and speaker rename registry."""

from __future__ import annotations

import copy
import threading


class MeetingRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._meetings: dict[str, dict] = {}

    def register(
        self,
        meeting_id: str,
        *,
        filename: str,
        segments: list[dict],
        speakers: list[dict],
    ) -> dict:
        stored_segments = []
        for segment in segments:
            item = copy.deepcopy(segment)
            item.pop("audio_wav_base64", None)
            stored_segments.append(item)
        payload = {
            "meeting_id": meeting_id,
            "filename": filename,
            "speakers": copy.deepcopy(speakers),
            "segments": stored_segments,
        }
        with self._lock:
            self._meetings[meeting_id] = payload
        return copy.deepcopy(payload)

    def get(self, meeting_id: str) -> dict | None:
        with self._lock:
            meeting = self._meetings.get(meeting_id)
            return copy.deepcopy(meeting) if meeting is not None else None

    def rename(self, meeting_id: str, speaker_id: str, name: str) -> dict | None:
        clean_name = " ".join(str(name).split())
        if not clean_name or len(clean_name) > 80:
            raise ValueError("speaker name must contain 1 to 80 characters")
        with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                return None
            matched = False
            for speaker in meeting["speakers"]:
                if speaker["id"] == speaker_id:
                    speaker["name"] = clean_name
                    matched = True
            if not matched:
                return None
            for segment in meeting["segments"]:
                if segment.get("speaker_id") == speaker_id:
                    segment["speaker_name"] = clean_name
            return copy.deepcopy(meeting)
