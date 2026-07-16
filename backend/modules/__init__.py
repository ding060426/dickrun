"""
会悟 - 智能会议/课堂语音认知系统
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

from .text_post_processor import (
    process_asr_text,
    process_transcript_segments,
    remove_fillers,
    restore_punctuation,
    force_split_long_sentence,
    normalize_text,
)
