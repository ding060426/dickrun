"""Stateful microphone audio pipeline.

The WebSocket layer only deals with protocol messages.  This module owns PCM
validation, VAD gating, pre-roll, and ASR utterance boundaries so the same
behavior can be exercised without a browser or a network connection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .asr_engine import ASRResult


class LiveAudioProtocolError(ValueError):
    """Raised when a microphone PCM frame violates the public wire contract."""


@dataclass(frozen=True)
class VadState:
    is_speech: bool
    speech_started: bool = False
    speech_ended: bool = False


class VoiceActivityDetector(Protocol):
    def push(self, samples: np.ndarray) -> VadState: ...


class AlwaysActiveVad:
    """Safe fallback that leaves utterance finalization to ASR/stream stop."""

    provider_name = "asr-endpoint-fallback"

    def __init__(self):
        self._started = False

    def push(self, samples: np.ndarray) -> VadState:
        started = not self._started
        self._started = True
        return VadState(is_speech=True, speech_started=started)


class SherpaSileroVad:
    """Streaming Silero VAD adapted from LocalMeet's sherpa-onnx pipeline."""

    provider_name = "sherpa-silero-vad"
    WINDOW_SIZE = 512

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.2,
        num_threads: int = 1,
    ):
        import sherpa_onnx

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {self.model_path}")
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(self.model_path),
                threshold=threshold,
                min_silence_duration=min_silence_duration,
                min_speech_duration=min_speech_duration,
                window_size=self.WINDOW_SIZE,
                max_speech_duration=60,
            ),
            sample_rate=LiveAudioSession.SAMPLE_RATE,
            num_threads=num_threads,
            provider="cpu",
        )
        self._detector = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=120,
        )
        self._pending = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> VadState:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate((self._pending, samples))
        speech_started = False
        speech_ended = False

        while len(self._pending) >= self.WINDOW_SIZE:
            window = self._pending[: self.WINDOW_SIZE]
            self._pending = self._pending[self.WINDOW_SIZE :]
            was_speech = self._detector.is_speech_detected()
            self._detector.accept_waveform(window)
            is_speech = self._detector.is_speech_detected()
            speech_started = speech_started or (not was_speech and is_speech)
            speech_ended = speech_ended or (was_speech and not is_speech)
            speech_ended = self._drain_segments() or speech_ended

        return VadState(
            is_speech=self._detector.is_speech_detected(),
            speech_started=speech_started,
            speech_ended=speech_ended,
        )

    def _drain_segments(self) -> bool:
        detected = False
        while not self._detector.empty():
            detected = True
            self._detector.pop()
        return detected


def find_silero_vad_model(model_dir: Path | None = None) -> Path | None:
    """Find a deployed model, with a development fallback to LocalMeet."""
    configured = os.getenv("DITING_SILERO_VAD_PATH", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))
    if model_dir is not None:
        candidates.append(Path(model_dir) / "silero_vad.onnx")
    package_model = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
    candidates.append(package_model)
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "LocalMeet_AI_Demo"
        / "data"
        / "models"
        / "audio-source"
        / "silero_vad.onnx"
    )
    return next((path for path in candidates if path.is_file()), None)


def create_live_vad(model_dir: Path | None = None) -> VoiceActivityDetector:
    model_path = find_silero_vad_model(model_dir)
    if model_path is None:
        return AlwaysActiveVad()
    return SherpaSileroVad(model_path)


class LiveAudioSession:
    """Convert 16 kHz mono PCM frames into partial and final ASR results."""

    SAMPLE_RATE = 16000
    MAX_FRAME_BYTES = 512 * 1024

    def __init__(
        self,
        engine,
        *,
        vad: VoiceActivityDetector,
        pre_roll_ms: int = 200,
    ):
        if pre_roll_ms < 0:
            raise ValueError("pre_roll_ms must be non-negative")
        self.engine = engine
        self.vad = vad
        self._pre_roll_limit = round(self.SAMPLE_RATE * pre_roll_ms / 1000)
        self._pre_roll: list[np.ndarray] = []
        self._pre_roll_samples = 0
        self._speech_active = False
        self._closed = False
        self._last_partial_text = ""
        self.received_samples = 0
        self.forwarded_samples = 0
        self.engine.start_session()

    def push_pcm_s16le(self, payload: bytes) -> list[ASRResult]:
        if self._closed:
            raise LiveAudioProtocolError("live audio session is already closed")
        if not payload or len(payload) % 2:
            raise LiveAudioProtocolError("PCM frame must contain complete int16 samples")
        if len(payload) > self.MAX_FRAME_BYTES:
            raise LiveAudioProtocolError("PCM frame is too large")

        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        self.received_samples += len(samples)
        state = self.vad.push(samples)

        if not self._speech_active:
            if not (state.speech_started or state.is_speech):
                self._remember_pre_roll(samples)
                return []
            samples = self._take_pre_roll(samples)
            self._speech_active = True

        events: list[ASRResult] = []
        result = self.engine.process_chunk(samples, sample_rate=self.SAMPLE_RATE)
        self.forwarded_samples += len(samples)
        if result and result.text.strip() and result.text != self._last_partial_text:
            events.append(result)
            self._last_partial_text = result.text

        if state.speech_ended:
            final = self.engine.finalize_utterance(reset_stream=True)
            if final and final.text.strip():
                events.append(final)
            self._speech_active = False
            self._last_partial_text = ""

        return events

    def finish(self) -> list[ASRResult]:
        if self._closed:
            return []
        self._closed = True
        self.engine.end_session()
        final = self.engine._finalize_results()
        if final and final.text.strip():
            return [final]
        return []

    def _remember_pre_roll(self, samples: np.ndarray) -> None:
        if self._pre_roll_limit <= 0:
            return
        self._pre_roll.append(np.array(samples, copy=True))
        self._pre_roll_samples += len(samples)
        while self._pre_roll and self._pre_roll_samples > self._pre_roll_limit:
            excess = self._pre_roll_samples - self._pre_roll_limit
            first = self._pre_roll[0]
            if len(first) <= excess:
                self._pre_roll.pop(0)
                self._pre_roll_samples -= len(first)
            else:
                self._pre_roll[0] = first[excess:]
                self._pre_roll_samples -= excess

    def _take_pre_roll(self, samples: np.ndarray) -> np.ndarray:
        if not self._pre_roll:
            return samples
        combined = np.concatenate((*self._pre_roll, samples))
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        return combined
