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

from .text_post_processor import (
    process_asr_text,
    process_transcript_segments,
    remove_fillers,
    restore_punctuation,
    force_split_long_sentence,
    normalize_text,
)

from .domain_taxonomy import (
    DOMAIN_TAXONOMY,
    STOP_WORDS,
    get_domain_keywords,
    get_all_domains,
    get_sub_domains,
    match_domain,
    merge_all_domain_keywords,
)

from .llm_client import (
    LLMClient,
    GPT55Client,
    DeepSeekClient,
    QwenClient,
    OpenAIClient,
    MockClient,
    get_llm_client,
    reset_llm_client,
)

from .speaker_diarization import (
    SpeakerIdentifier,
    SpeakerDiarizer,
    extract_acoustic_embedding,
    create_speaker_pipeline,
)

from .hotword_engine import (
    HotwordExtractor,
    PinyinFuzzyMatcher,
    extract_and_match,
)

from .asr_optimizer import (
    ASROptimizer,
    ASROptimizerReport,
)

from .cognitive_engine import (
    ContentPredictor,
    DomainInferrer,
    MeetingSummarizer,
    create_cognitive_pipeline,
)

try:
    from .hotword_processor import HotwordProcessor
except Exception:  # pragma: no cover - optional during partial installs
    HotwordProcessor = None

__all__ = [
    "estimate_snr",
    "estimate_rt60",
    "classify_scene",
    "acoustic_quality_score",
    "SpeakerEmbedder",
    "HotwordCorrector",
    "LogicValidator",
    "UncertaintyEstimator",
    "MeetingProcessor",
    "process_asr_text",
    "process_transcript_segments",
    "remove_fillers",
    "restore_punctuation",
    "force_split_long_sentence",
    "normalize_text",
    "DOMAIN_TAXONOMY",
    "STOP_WORDS",
    "get_domain_keywords",
    "get_all_domains",
    "get_sub_domains",
    "match_domain",
    "merge_all_domain_keywords",
    "LLMClient",
    "GPT55Client",
    "DeepSeekClient",
    "QwenClient",
    "OpenAIClient",
    "MockClient",
    "get_llm_client",
    "reset_llm_client",
    "SpeakerIdentifier",
    "SpeakerDiarizer",
    "extract_acoustic_embedding",
    "create_speaker_pipeline",
    "HotwordExtractor",
    "PinyinFuzzyMatcher",
    "extract_and_match",
    "ASROptimizer",
    "ASROptimizerReport",
    "ContentPredictor",
    "DomainInferrer",
    "MeetingSummarizer",
    "create_cognitive_pipeline",
    "HotwordProcessor",
]
