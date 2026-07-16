"""Lightweight realtime VAD for HUIWU live sessions."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from modules.pcm_utils import rms_level, sanitize_float32
except Exception:  # pragma: no cover
    from pcm_utils import rms_level, sanitize_float32


@dataclass
class VADState:
    state: str = "silence"
    rms: float = 0.0
    noise_floor: float = 0.003
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    endpoint: bool = False


class RealtimeVAD:
    """Energy VAD with adaptive noise floor and endpoint grace."""

    def __init__(
        self,
        sample_rate: int = 16000,
        start_ratio: float = 3.0,
        end_ratio: float = 1.8,
        min_speech_ms: int = 180,
        endpoint_silence_ms: int = 800,
        max_utterance_ms: int = 30000,
    ):
        self.sample_rate = int(sample_rate or 16000)
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)
        self.min_speech_ms = int(min_speech_ms)
        self.endpoint_silence_ms = int(endpoint_silence_ms)
        self.max_utterance_ms = int(max_utterance_ms)
        self.state = VADState()
        self._utterance_ms = 0.0

    def reset(self) -> None:
        self.state = VADState(noise_floor=self.state.noise_floor)
        self._utterance_ms = 0.0

    def update(self, audio, sample_rate: int | None = None) -> VADState:
        data = sanitize_float32(audio)
        sr = int(sample_rate or self.sample_rate)
        duration_ms = (len(data) / max(1, sr)) * 1000.0
        rms = rms_level(data)
        st = self.state
        st.rms = rms
        st.endpoint = False

        # Track the floor mostly while silent; slow update prevents speech from
        # immediately lifting the floor.
        if st.state == "silence" or rms < st.noise_floor * self.end_ratio:
            st.noise_floor = max(0.0005, st.noise_floor * 0.95 + rms * 0.05)

        start_th = max(0.006, st.noise_floor * self.start_ratio)
        end_th = max(0.004, st.noise_floor * self.end_ratio)
        is_speech = rms >= start_th if st.state == "silence" else rms >= end_th

        if is_speech:
            st.speech_ms += duration_ms
            st.silence_ms = 0.0
            self._utterance_ms += duration_ms
            if st.speech_ms >= self.min_speech_ms:
                st.state = "speech"
        else:
            st.silence_ms += duration_ms
            if st.state == "speech":
                self._utterance_ms += duration_ms
                if st.silence_ms >= self.endpoint_silence_ms or self._utterance_ms >= self.max_utterance_ms:
                    st.endpoint = True
                    st.state = "silence"
                    st.speech_ms = 0.0
                    st.silence_ms = 0.0
                    self._utterance_ms = 0.0
            else:
                st.state = "silence"
                st.speech_ms = 0.0
        return VADState(
            state=st.state,
            rms=st.rms,
            noise_floor=st.noise_floor,
            speech_ms=st.speech_ms,
            silence_ms=st.silence_ms,
            endpoint=st.endpoint,
        )
