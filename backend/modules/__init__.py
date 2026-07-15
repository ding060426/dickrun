"""
谛听 DiTing - 智能会议/课堂语音认知系统
Backend Modules
"""

from .audio_processor import (
    estimate_snr,
    estimate_rt60,
    classify_scene,
    acoustic_quality_score,
    SpeakerEmbedder,
    HotwordCorrector,
    LogicValidator,
    UncertaintyEstimator,
    MeetingProcessor,
)
