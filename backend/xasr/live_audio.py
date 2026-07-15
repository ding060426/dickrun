"""Stateful microphone audio pipeline.

The WebSocket layer only deals with protocol messages.  This module owns PCM
validation, VAD gating, pre-roll, and ASR utterance boundaries so the same
behavior can be exercised without a browser or a network connection.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .contracts import ASRResult
from .recording import LiveRecording, RecordingResult


DEFAULT_ENDPOINT_GRACE_MS = 800


@dataclass(frozen=True)
class LiveAudioProfile:
    name: str
    pre_roll_ms: int
    endpoint_grace_ms: int
    tail_pad_ms: int
    vad_threshold: float
    vad_min_silence: float
    vad_min_speech: float


_LIVE_AUDIO_PROFILES = {
    "meeting": LiveAudioProfile("meeting", 700, 800, 1000, 0.5, 0.5, 0.2),
    "dictation": LiveAudioProfile("dictation", 300, 200, 1000, 0.5, 0.35, 0.1),
    "oncall": LiveAudioProfile("oncall", 700, 600, 1000, 0.5, 0.5, 0.2),
}


def get_live_audio_profile(name: str | None = None) -> LiveAudioProfile:
    selected = (name or os.getenv("DITING_LIVE_PROFILE", "meeting")).strip().lower()
    try:
        base = _LIVE_AUDIO_PROFILES[selected]
    except KeyError as error:
        choices = ", ".join(_LIVE_AUDIO_PROFILES)
        raise ValueError(f"unknown live profile {selected!r}; choose one of: {choices}") from error
    return LiveAudioProfile(
        name=base.name,
        pre_roll_ms=_env_int("DITING_LIVE_PRE_ROLL_MS", base.pre_roll_ms),
        endpoint_grace_ms=_env_int(
            "DITING_LIVE_ENDPOINT_GRACE_MS",
            base.endpoint_grace_ms,
        ),
        tail_pad_ms=_env_int("DITING_LIVE_TAIL_PAD_MS", base.tail_pad_ms),
        vad_threshold=_env_float("DITING_LIVE_VAD_THRESHOLD", base.vad_threshold),
        vad_min_silence=_env_float(
            "DITING_LIVE_VAD_MIN_SILENCE",
            base.vad_min_silence,
        ),
        vad_min_speech=_env_float(
            "DITING_LIVE_VAD_MIN_SPEECH",
            base.vad_min_speech,
        ),
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def get_live_endpoint_grace_ms() -> int:
    """Return the configurable post-VAD silence grace period."""
    return get_live_audio_profile().endpoint_grace_ms


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


class EnergyVad:
    """Dependency-free streaming VAD used when no neural VAD is deployed."""

    provider_name = "energy-vad"

    def __init__(
        self,
        *,
        threshold: float = 0.012,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.2,
        sample_rate: int = 16000,
    ):
        self._threshold = max(0.0001, float(threshold))
        self._min_silence_samples = round(sample_rate * min_silence_duration)
        self._min_speech_samples = round(sample_rate * min_speech_duration)
        self._speech_samples = 0
        self._silence_samples = 0
        self._active = False

    def push(self, samples: np.ndarray) -> VadState:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        voiced = rms >= self._threshold

        if not self._active:
            self._speech_samples = self._speech_samples + len(samples) if voiced else 0
            if self._speech_samples >= self._min_speech_samples:
                self._active = True
                self._silence_samples = 0
                return VadState(is_speech=True, speech_started=True)
            return VadState(is_speech=False)

        if voiced:
            self._silence_samples = 0
            return VadState(is_speech=True)

        self._silence_samples += len(samples)
        if self._silence_samples >= self._min_silence_samples:
            self._active = False
            self._speech_samples = 0
            self._silence_samples = 0
            return VadState(is_speech=False, speech_ended=True)
        return VadState(is_speech=True)


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


def create_live_vad(
    model_dir: Path | None = None,
    *,
    profile: LiveAudioProfile | None = None,
) -> VoiceActivityDetector:
    selected = profile or get_live_audio_profile()
    requested = os.getenv("DITING_LIVE_VAD", "auto").strip().lower()
    model_path = find_silero_vad_model(model_dir)
    if requested != "energy" and model_path is not None:
        return SherpaSileroVad(
            model_path,
            threshold=selected.vad_threshold,
            min_silence_duration=selected.vad_min_silence,
            min_speech_duration=selected.vad_min_speech,
        )
    return EnergyVad(
        threshold=_env_float("DITING_LIVE_ENERGY_THRESHOLD", 0.012),
        min_silence_duration=selected.vad_min_silence,
        min_speech_duration=selected.vad_min_speech,
    )


class LiveAudioSession:
    """Convert PCM frames into ASR results with resumable VAD boundaries.

    Silero's speech-end signal is treated as a candidate boundary. The active
    ASR stream stays open for a short grace period so a natural thinking pause
    can resume without losing the decoder context.
    """

    SAMPLE_RATE = 16000
    MAX_FRAME_BYTES = 512 * 1024
    WIRE_MAGIC = b"DTP2"
    WIRE_HEADER = struct.Struct("<4sI")

    def __init__(
        self,
        engine,
        *,
        vad: VoiceActivityDetector,
        pre_roll_ms: int = 200,
        endpoint_grace_ms: int = DEFAULT_ENDPOINT_GRACE_MS,
        tail_pad_ms: int = 1000,
        recording: LiveRecording | None = None,
    ):
        if pre_roll_ms < 0:
            raise ValueError("pre_roll_ms must be non-negative")
        if endpoint_grace_ms < 0:
            raise ValueError("endpoint_grace_ms must be non-negative")
        if tail_pad_ms < 0:
            raise ValueError("tail_pad_ms must be non-negative")
        self.engine = engine
        self.vad = vad
        self._pre_roll_limit = round(self.SAMPLE_RATE * pre_roll_ms / 1000)
        self._endpoint_grace_samples = round(
            self.SAMPLE_RATE * endpoint_grace_ms / 1000
        )
        self._tail_pad_ms = tail_pad_ms
        self._recording = recording
        self.recording_result: RecordingResult | None = None
        self._pre_roll: list[np.ndarray] = []
        self._pre_roll_samples = 0
        self._speech_active = False
        self._pending_endpoint_samples: int | None = None
        self._closed = False
        self._last_partial_text = ""
        self.received_samples = 0
        self.forwarded_samples = 0
        self.received_frames = 0
        self.partial_results = 0
        self.final_results = 0
        self._started_at = time.perf_counter()
        self._first_partial_at: float | None = None
        self.last_sequence = -1
        self.dropped_frames = 0
        self.engine.start_session()

    def push_binary_frame(self, payload: bytes) -> list[ASRResult]:
        """Accept a versioned DTP2 frame or a legacy raw PCM frame."""
        if payload.startswith(self.WIRE_MAGIC):
            if len(payload) <= self.WIRE_HEADER.size:
                raise LiveAudioProtocolError("DTP2 frame is missing PCM payload")
            _, sequence = self.WIRE_HEADER.unpack_from(payload)
            if sequence <= self.last_sequence:
                raise LiveAudioProtocolError("DTP2 sequence must be strictly increasing")
            if self.last_sequence >= 0 and sequence > self.last_sequence + 1:
                self.dropped_frames += sequence - self.last_sequence - 1
            self.last_sequence = sequence
            payload = payload[self.WIRE_HEADER.size :]
        return self.push_pcm_s16le(payload)

    def push_pcm_s16le(self, payload: bytes) -> list[ASRResult]:
        if self._closed:
            raise LiveAudioProtocolError("live audio session is already closed")
        if not payload or len(payload) % 2:
            raise LiveAudioProtocolError("PCM frame must contain complete int16 samples")
        if len(payload) > self.MAX_FRAME_BYTES:
            raise LiveAudioProtocolError("PCM frame is too large")

        if self._recording is not None:
            self._recording.append_pcm_s16le(payload)

        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        self.received_frames += 1
        self.received_samples += len(samples)
        state = self.vad.push(samples)

        if self._pending_endpoint_samples is not None and (
            state.speech_started or state.is_speech
        ):
            self._pending_endpoint_samples = None

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
            self.partial_results += 1
            if self._first_partial_at is None:
                self._first_partial_at = time.perf_counter()
            self._last_partial_text = result.text

        if state.speech_ended:
            self._pending_endpoint_samples = 0
        elif self._pending_endpoint_samples is not None and not state.is_speech:
            self._pending_endpoint_samples += len(samples)

        if (
            self._pending_endpoint_samples is not None
            and self._pending_endpoint_samples >= self._endpoint_grace_samples
        ):
            final = self.engine.finalize_utterance(
                reset_stream=True,
                tail_pad_ms=self._tail_pad_ms,
            )
            if final and final.text.strip():
                events.append(final)
                self.final_results += 1
            self._speech_active = False
            self._pending_endpoint_samples = None
            self._last_partial_text = ""

        return events

    def finish(self) -> list[ASRResult]:
        if self._closed:
            return []
        self._closed = True
        try:
            final = self.engine.finalize_utterance(
                reset_stream=False,
                tail_pad_ms=self._tail_pad_ms,
            )
        finally:
            if self._recording is not None:
                self.recording_result = self._recording.finalize()
        if final and final.text.strip():
            self.final_results += 1
            return [final]
        return []

    def metrics(self) -> dict:
        elapsed_ms = round((time.perf_counter() - self._started_at) * 1000, 1)
        first_partial_ms = None
        if self._first_partial_at is not None:
            first_partial_ms = round(
                (self._first_partial_at - self._started_at) * 1000,
                1,
            )
        return {
            "protocol_version": 2 if self.last_sequence >= 0 else 1,
            "received_frames": self.received_frames,
            "received_samples": self.received_samples,
            "forwarded_samples": self.forwarded_samples,
            "last_sequence": self.last_sequence,
            "dropped_frames": self.dropped_frames,
            "partial_results": self.partial_results,
            "final_results": self.final_results,
            "first_partial_ms": first_partial_ms,
            "elapsed_ms": elapsed_ms,
        }

    def abort(self) -> Path | None:
        """Stop without final ASR and preserve any partial recording for recovery."""
        if self._closed:
            if self.recording_result is not None:
                return self.recording_result.path
            return None
        self._closed = True
        self.engine.end_session()
        return self._recording.abort() if self._recording is not None else None

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
