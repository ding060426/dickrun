"""Configuration profiles for X-ASR latency/accuracy trade-offs."""

from __future__ import annotations

import os


ASR_CHUNK_PROFILES = {
    "low-latency": 160,
    "balanced": 480,
    "meeting": 960,
    "quality": 1920,
}


def resolve_asr_profile(value: str | None = None) -> tuple[str, int]:
    profile = (value or os.getenv("DITING_ASR_PROFILE", "low-latency")).strip().lower()
    aliases = {
        "live": "low-latency",
        "fast": "low-latency",
        "default": "low-latency",
    }
    profile = aliases.get(profile, profile)
    try:
        return profile, ASR_CHUNK_PROFILES[profile]
    except KeyError as error:
        choices = ", ".join(ASR_CHUNK_PROFILES)
        raise ValueError(f"unknown ASR profile {profile!r}; choose one of: {choices}") from error
