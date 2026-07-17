"""X-ASR integration with optional, lazily imported model runtime."""

__all__ = ["ASRResult", "SherpaStreamingASR", "XASREngine"]


def __getattr__(name: str):
    if name == "ASRResult":
        from .contracts import ASRResult

        return ASRResult
    if name == "SherpaStreamingASR":
        from .sherpa_streaming_infer import SherpaStreamingASR

        return SherpaStreamingASR
    if name == "XASREngine":
        from .asr_engine import XASREngine

        return XASREngine
    raise AttributeError(name)
