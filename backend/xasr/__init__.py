"""
X-ASR 集成模块
基于 X-ASR (https://github.com/Gilgamesh-J/X-ASR) 的 sherpa-onnx 推理引擎
"""

from .sherpa_streaming_infer import SherpaStreamingASR
from .asr_engine import XASREngine, ASRResult
