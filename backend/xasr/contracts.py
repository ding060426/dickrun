"""Dependency-free contracts shared by X-ASR orchestration modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ASRResult:
    """Recognition result exposed to transports, tests, and enhancement modules."""

    text: str = ""
    raw_text: str = ""
    is_partial: bool = False
    is_final: bool = False
    timestamp: float = 0.0
    start_sec: float = 0.0
    end_sec: float = 0.0
    audio_data: Optional[np.ndarray] = field(default=None, repr=False)
    asr_confidence: float = 0.8
    snr_db: float = 25.0
    rt60: float = 0.3
    quality_score: float = 0.85
    quality_label: str = "high"
    corrections: list[dict] = field(default_factory=list)
    logic_flags: list[dict] = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    terms: list[str] = field(default_factory=list)
    data_points: list[dict] = field(default_factory=list)
    uncertain_spans: list[dict] = field(default_factory=list)
