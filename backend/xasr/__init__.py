"""X-ASR integration with optional, lazily imported model runtime."""

from .contracts import ASRResult

__all__ = ["ASRResult", "SherpaStreamingASR", "XASREngine"]


def __getattr__(name: str):
    if name == "SherpaStreamingASR":
        from .sherpa_streaming_infer import SherpaStreamingASR

        return SherpaStreamingASR
    if name == "XASREngine":
        from .asr_engine import XASREngine

        return XASREngine
    raise AttributeError(name)
