"""Live session service boundary.

LiveAudioSession construction is still performed by backend.main so the current
public /ws/live contract remains stable. This file exists as the service-layer
home for the next mechanical migration.
"""

from __future__ import annotations


def asr_engine_label(engine) -> str:
    return str(getattr(engine, "engine_name", "X-ASR (sherpa-onnx zipformer2 v2.0)"))
