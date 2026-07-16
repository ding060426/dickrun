"""DTP2 realtime audio protocol helpers.

The teammate branch uses a compact binary protocol with a DTP2 magic header.
This module keeps that path optional and compatible with DiTing's older JSON
base64 Float32 WebSocket protocol.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from modules.pcm_utils import int16_pcm_to_float32, sanitize_float32
except Exception:  # pragma: no cover
    from pcm_utils import int16_pcm_to_float32, sanitize_float32


WIRE_MAGIC = b"DTP2"
WIRE_HEADER = struct.Struct("<4sI")  # magic, sequence


@dataclass
class RealtimePacket:
    kind: str
    action: str = ""
    audio: Optional[np.ndarray] = None
    sample_rate: int = 16000
    channels: int = 1
    fmt: str = "float32"
    seq: Optional[int] = None
    protocol: str = "json"
    raw: Optional[dict[str, Any]] = None
    error: str = ""
    rnnoise_enabled: Optional[bool] = None


def _parse_optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "open", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "close", "disabled"}:
        return False
    return None


def _control_action(msg: dict[str, Any]) -> str:
    typ = str(msg.get("type") or msg.get("action") or "").lower()
    if typ in {"dtp2.start", "start", "configure"}:
        return "start"
    if typ in {"dtp2.stop", "stop"}:
        return "stop"
    if typ == "process_chunk":
        return "audio"
    return typ or "control"


def parse_text_message(text: str) -> RealtimePacket:
    """Parse a JSON text WebSocket message into a packet."""
    try:
        msg = json.loads(text)
    except Exception as exc:
        return RealtimePacket(kind="error", action="parse", error=f"Invalid JSON: {exc}")
    if not isinstance(msg, dict):
        return RealtimePacket(kind="error", action="parse", error="JSON message must be an object")

    action = _control_action(msg)
    if action == "audio":
        audio_b64 = msg.get("audio", "")
        if not audio_b64:
            return RealtimePacket(kind="error", action="audio", error="Missing audio payload", raw=msg)
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            return RealtimePacket(
                kind="audio",
                action="audio",
                audio=sanitize_float32(audio),
                sample_rate=int(msg.get("sample_rate") or 16000),
                channels=int(msg.get("channels") or 1),
                fmt="float32",
                protocol="json",
                raw=msg,
                rnnoise_enabled=_parse_optional_bool(msg.get("rnnoise")),
            )
        except Exception as exc:
            return RealtimePacket(kind="error", action="audio", error=f"Invalid audio payload: {exc}", raw=msg)

    return RealtimePacket(
        kind="control",
        action=action,
        sample_rate=int(msg.get("sample_rate") or 16000),
        channels=int(msg.get("channels") or 1),
        fmt=str(msg.get("format") or msg.get("sample_format") or "pcm_s16le"),
        protocol="dtp2" if str(msg.get("type", "")).lower().startswith("dtp2") else "json",
        raw=msg,
        rnnoise_enabled=_parse_optional_bool(msg.get("rnnoise")),
    )


def parse_binary_message(data: bytes) -> RealtimePacket:
    """Parse a DTP2 binary audio frame."""
    if not data:
        return RealtimePacket(kind="error", action="audio", error="Empty binary frame", protocol="dtp2")
    if len(data) < WIRE_HEADER.size:
        return RealtimePacket(kind="error", action="audio", error="DTP2 frame too short", protocol="dtp2")
    try:
        magic, seq = WIRE_HEADER.unpack(data[:WIRE_HEADER.size])
    except Exception as exc:
        return RealtimePacket(kind="error", action="audio", error=f"Invalid DTP2 header: {exc}", protocol="dtp2")
    if magic != WIRE_MAGIC:
        return RealtimePacket(kind="error", action="audio", error="Invalid DTP2 magic", protocol="dtp2")
    payload = data[WIRE_HEADER.size:]
    audio = int16_pcm_to_float32(payload)
    return RealtimePacket(
        kind="audio",
        action="audio",
        audio=audio,
        sample_rate=16000,
        channels=1,
        fmt="pcm_s16le",
        seq=int(seq),
        protocol="dtp2",
    )
