"""
VAD Manager: Unified voice activity detection with three-tier fallback.

Tier 1: FireRedVAD (DFSMN neural, 97.57% F1, streaming-capable) — best quality
Tier 2: Silero VAD (sherpa-onnx neural, 95.95% F1) — good quality, zero extra deps
Tier 3: Energy VAD (RMS threshold) — basic fallback, zero dependencies

All three expose the same interface: segment_audio(audio, sr) → List[(start, end)]
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger("diting")

# Model paths
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xasr", "models")
_FIRERED_MODEL_DIR = os.path.join(_MODELS_DIR, "firered_vad", "Stream-VAD")
_SILERO_MODEL_PATH = os.path.join(_MODELS_DIR, "silero_vad.onnx")


# ============================================================
# Tier 1: FireRedVAD (DFSMN streaming VAD)
# ============================================================

_firered_vad = None
_firered_loaded = False


def _get_firered_vad():
    """Lazy-load FireRedVAD model (singleton)."""
    global _firered_vad, _firered_loaded
    if _firered_loaded:
        return _firered_vad
    _firered_loaded = True

    if not os.path.isdir(_FIRERED_MODEL_DIR):
        logger.info(f"FireRedVAD model not found at {_FIRERED_MODEL_DIR}")
        return None

    try:
        import sys
        # Ensure torch is available
        try:
            import torch
        except ImportError:
            for p in ['C:\\pylibs']:
                if p not in sys.path:
                    sys.path.append(p)
            try:
                import torch
            except ImportError:
                logger.warning("torch not available for FireRedVAD")
                return None

        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        cfg = FireRedStreamVadConfig(
            speech_threshold=0.5,
            min_speech_frame=20,       # 200ms min speech
            min_silence_frame=50,     # 500ms silence → sentence break
            smooth_window_size=5,
            pad_start_frame=5,        # 50ms preroll
        )
        _firered_vad = FireRedStreamVad.from_pretrained(_FIRERED_MODEL_DIR, cfg)
        logger.info(f"FireRedVAD loaded (DFSMN, 97.57% F1, threshold=0.5)")
        return _firered_vad
    except Exception as e:
        logger.warning(f"FireRedVAD load failed: {e}")
        return None


def firered_vad_segment(
    audio: np.ndarray,
    sample_rate: int,
    max_duration: float = 30.0,
    min_segment_duration: float = 0.5,
    pre_padding_ms: float = 200,
    post_padding_ms: float = 200,
) -> Optional[List[Tuple[float, float]]]:
    """Segment audio using FireRedVAD (neural, streaming-capable).

    Returns list of (start_sec, end_sec) or None if FireRedVAD unavailable.
    """
    vad = _get_firered_vad()
    if vad is None:
        return None

    try:
        vad.reset()
    except Exception:
        pass

    audio = audio.astype(np.float32)
    n_samples = len(audio)
    total_duration = n_samples / sample_rate

    # FireRedVAD expects int16 PCM
    i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    # detect_full processes the entire audio and returns timestamps
    frame_results, result = vad.detect_full(i16)
    raw_segments = result.get('timestamps', [])

    if not raw_segments:
        logger.warning(f"FireRedVAD: no speech detected in {total_duration:.1f}s")
        return [(0.0, total_duration)]

    # Apply padding
    pre_pad = pre_padding_ms / 1000
    post_pad = post_padding_ms / 1000
    padded = []
    for s, e in raw_segments:
        s = max(0, s - pre_pad)
        e = min(total_duration, e + post_pad)
        padded.append((s, e))

    # Merge adjacent segments (within 0.3s)
    merged = []
    for seg in sorted(padded):
        if merged and seg[0] <= merged[-1][1] + 0.3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    # Apply max_duration cap
    final = _apply_max_duration(merged, max_duration, min_segment_duration, total_duration)

    logger.info(f"FireRedVAD: detected {len(final)} segments in {total_duration:.1f}s audio")
    return final


# ============================================================
# Tier 2: Silero VAD (sherpa-onnx)
# ============================================================

_silero_vad = None
_silero_loaded = False


def _get_silero_vad():
    """Lazy-load silero VAD model (singleton)."""
    global _silero_vad, _silero_loaded
    if _silero_loaded:
        return _silero_vad
    _silero_loaded = True

    if not os.path.isfile(_SILERO_MODEL_PATH):
        logger.info(f"Silero VAD model not found at {_SILERO_MODEL_PATH}")
        return None

    try:
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = _SILERO_MODEL_PATH
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.5
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.window_size = 512
        _silero_vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        logger.info(f"Silero VAD loaded (neural, threshold=0.5)")
        return _silero_vad
    except Exception as e:
        logger.warning(f"Silero VAD load failed: {e}")
        return None


def silero_vad_segment(
    audio: np.ndarray,
    sample_rate: int,
    max_duration: float = 30.0,
    min_segment_duration: float = 0.5,
    pre_padding_ms: float = 200,
    post_padding_ms: float = 200,
) -> Optional[List[Tuple[float, float]]]:
    """Segment audio using Silero VAD (sherpa-onnx).

    Returns list of (start_sec, end_sec) or None if Silero unavailable.
    """
    vad = _get_silero_vad()
    if vad is None:
        return None

    try:
        vad.reset()
    except Exception:
        pass

    audio = audio.astype(np.float32)
    n_samples = len(audio)
    total_duration = n_samples / sample_rate
    window_size = 512

    # Feed audio window by window, collect segments
    segments = []
    for i in range(0, n_samples, window_size):
        chunk = audio[i:i + window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        vad.accept_waveform(chunk)

        while not vad.empty():
            seg = vad.front
            vad.pop()
            seg_start = seg.start / sample_rate
            if len(seg.samples) > 0:
                seg_end = seg_start + len(seg.samples) / sample_rate
            else:
                seg_end = (i + window_size) / sample_rate
            segments.append((seg_start, seg_end))

    vad.flush()
    while not vad.empty():
        seg = vad.front
        vad.pop()
        seg_start = seg.start / sample_rate
        if len(seg.samples) > 0:
            seg_end = seg_start + len(seg.samples) / sample_rate
        else:
            seg_end = total_duration
        segments.append((seg_start, seg_end))

    if not segments:
        logger.warning(f"Silero VAD: no speech detected in {total_duration:.1f}s")
        return [(0.0, total_duration)]

    # Apply padding
    pre_pad = pre_padding_ms / 1000
    post_pad = post_padding_ms / 1000
    padded = [(max(0, s - pre_pad), min(total_duration, e + post_pad)) for s, e in segments]

    # Merge adjacent segments (within 0.5s)
    merged = []
    for seg in sorted(padded):
        if merged and seg[0] <= merged[-1][1] + 0.5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    # Apply max_duration cap
    final = _apply_max_duration(merged, max_duration, min_segment_duration, total_duration)

    logger.info(f"Silero VAD: detected {len(final)} segments in {total_duration:.1f}s audio")
    return final


# ============================================================
# Tier 3: Energy VAD (RMS threshold fallback)
# ============================================================

def energy_vad_segment(
    audio: np.ndarray,
    sample_rate: int,
    max_duration: float = 30.0,
    min_segment_duration: float = 0.5,
    energy_threshold_ratio: float = 0.06,
    min_silence_frames: int = 50,
    pre_padding_ms: float = 200,
    post_padding_ms: float = 200,
) -> List[Tuple[float, float]]:
    """Segment audio using simple energy-based VAD (zero dependencies)."""
    frame_ms = 25
    hop_ms = 10
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = int(sample_rate * hop_ms / 1000)
    min_speech_frames = 15

    audio = np.asarray(audio, dtype=np.float32)
    n_samples = len(audio)
    total_duration = n_samples / sample_rate

    if n_samples < frame_size:
        return [(0.0, total_duration)]

    # Compute frame energies
    energies = []
    for i in range(0, n_samples - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        energies.append(float(np.sqrt(np.mean(frame ** 2)) + 1e-9))

    if not energies:
        return [(0.0, total_duration)]

    # Dynamic threshold
    max_energy = max(energies)
    threshold = max_energy * energy_threshold_ratio

    # Detect speech/silence frames
    is_speech = [e > threshold for e in energies]

    # Find speech segments
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                in_speech = True
                speech_start = i * hop_ms / 1000
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    speech_end = (i - silence_count) * hop_ms / 1000
                    segments.append((speech_start, speech_end))
                    in_speech = False

    if in_speech:
        segments.append((speech_start, total_duration))

    if not segments:
        return [(0.0, total_duration)]

    # Apply padding
    pre_pad = pre_padding_ms / 1000
    post_pad = post_padding_ms / 1000
    padded = [(max(0, s - pre_pad), min(total_duration, e + post_pad)) for s, e in segments]

    # Merge adjacent segments (within 0.3s)
    merged = []
    for seg in sorted(padded):
        if merged and seg[0] <= merged[-1][1] + 0.3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    # Apply max_duration cap
    final = _apply_max_duration(merged, max_duration, min_segment_duration, total_duration)

    logger.info(f"Energy VAD: detected {len(final)} segments in {total_duration:.1f}s audio")
    return final


# ============================================================
# Shared utility: max_duration enforcement
# ============================================================

def _apply_max_duration(
    segments: List[Tuple[float, float]],
    max_duration: float,
    min_segment_duration: float,
    total_duration: float,
) -> List[Tuple[float, float]]:
    """Force-split segments longer than max_duration."""
    final = []
    for seg in segments:
        dur = seg[1] - seg[0]
        if dur > max_duration:
            n_sub = int(np.ceil(dur / max_duration))
            sub_dur = dur / n_sub
            for k in range(n_sub):
                sub_start = seg[0] + k * sub_dur
                sub_end = min(seg[1], seg[0] + (k + 1) * sub_dur)
                if sub_end - sub_start >= min_segment_duration:
                    final.append((sub_start, sub_end))
        elif dur >= min_segment_duration:
            final.append(seg)
    return final


# ============================================================
# Unified VAD Manager
# ============================================================

# Track which VAD is active
_active_vad = None  # "firered" / "silero" / "energy"


def summarize_segments(segments: List[Tuple[float, float]], total_duration: float) -> dict:
    """Return compact statistics for a list of VAD segments."""
    safe_segments = [(float(s), float(e)) for s, e in (segments or []) if e >= s]
    durations = [max(0.0, e - s) for s, e in safe_segments]
    speech_duration = sum(durations)
    coverage = speech_duration / total_duration if total_duration > 0 else 0.0
    return {
        "segments_count": len(safe_segments),
        "speech_duration_sec": round(speech_duration, 3),
        "speech_coverage_ratio": round(coverage, 4),
        "min_segment_duration": round(min(durations), 3) if durations else 0.0,
        "max_segment_duration": round(max(durations), 3) if durations else 0.0,
        "avg_segment_duration": round(speech_duration / len(durations), 3) if durations else 0.0,
        "is_full_audio_fallback": bool(
            len(safe_segments) == 1
            and total_duration > 0
            and safe_segments[0][0] <= 0.01
            and safe_segments[0][1] >= total_duration * 0.98
        ),
    }


def get_vad_info() -> dict:
    """Return info about available VAD backends."""
    global _active_vad
    info = {
        "firered_available": os.path.isdir(_FIRERED_MODEL_DIR),
        "silero_available": os.path.isfile(_SILERO_MODEL_PATH),
        "active_vad": _active_vad,
        "priority": ["firered", "silero", "energy"],
        "models_dir": os.path.abspath(_MODELS_DIR),
        "firered_model_dir": os.path.abspath(_FIRERED_MODEL_DIR),
        "silero_model_path": os.path.abspath(_SILERO_MODEL_PATH),
        "firered_loaded": _firered_loaded,
        "silero_loaded": _silero_loaded,
        "default_config": {
            "max_duration": 30.0,
            "min_segment_duration": 0.5,
            "pre_padding_ms": 200,
            "post_padding_ms": 200,
        },
    }
    return info


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_duration: float = 30.0,
    min_segment_duration: float = 0.5,
    pre_padding_ms: float = 200,
    post_padding_ms: float = 200,
) -> Tuple[List[Tuple[float, float]], str]:
    """
    Segment audio using the best available VAD.

    Priority: FireRedVAD → Silero VAD → Energy VAD

    Returns:
        (segments, vad_type): segments is list of (start_sec, end_sec),
                                vad_type is "firered" / "silero" / "energy"
    """
    global _active_vad

    # Try Tier 1: FireRedVAD
    result = firered_vad_segment(
        audio, sample_rate,
        max_duration=max_duration,
        min_segment_duration=min_segment_duration,
        pre_padding_ms=pre_padding_ms,
        post_padding_ms=post_padding_ms,
    )
    if result is not None:
        _active_vad = "firered"
        return result, "firered"

    # Try Tier 2: Silero VAD
    result = silero_vad_segment(
        audio, sample_rate,
        max_duration=max_duration,
        min_segment_duration=min_segment_duration,
        pre_padding_ms=pre_padding_ms,
        post_padding_ms=post_padding_ms,
    )
    if result is not None:
        _active_vad = "silero"
        return result, "silero"

    # Tier 3: Energy VAD (always works)
    _active_vad = "energy"
    result = energy_vad_segment(
        audio, sample_rate,
        max_duration=max_duration,
        min_segment_duration=min_segment_duration,
        pre_padding_ms=pre_padding_ms,
        post_padding_ms=post_padding_ms,
    )
    return result, "energy"


# ============================================================
# Streaming VAD for real-time mic input
# ============================================================

class StreamVADState:
    """Stateful streaming VAD for real-time microphone input.

    Wraps FireRedVAD's detect_chunk() or Silero's accept_waveform()
    for frame-by-frame processing.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = None
        self.vad = None
        self.in_speech = False
        self._pending_segments = []

        if backend in ("auto", "firered"):
            v = _get_firered_vad()
            if v is not None:
                self.backend = "firered"
                self.vad = v
                self._chunk_buf = np.zeros(0, dtype=np.float32)
                self._chunk_size = int(0.3 * 16000)  # 300ms chunks

        if self.backend is None and backend in ("auto", "silero"):
            v = _get_silero_vad()
            if v is not None:
                self.backend = "silero"
                self.vad = v

        if self.backend is None:
            self.backend = "energy"
            self.vad = _EnergyStreamVad()
            logger.info("StreamVAD: using energy VAD fallback")

    def reset(self):
        """Reset VAD state."""
        try:
            self.vad.reset()
        except Exception:
            pass
        self.in_speech = False
        self._pending_segments = []

    def accept_waveform(self, samples: np.ndarray) -> List[Tuple[float, float]]:
        """
        Feed audio samples to the VAD.

        Returns list of (start_sec, end_sec) for any completed segments.
        """
        samples = np.asarray(samples, dtype=np.float32)
        completed = []

        if self.backend == "firered":
            # FireRedVAD: accumulate chunks and call detect_chunk
            self._chunk_buf = np.concatenate([self._chunk_buf, samples])
            while len(self._chunk_buf) >= self._chunk_size:
                chunk = self._chunk_buf[:self._chunk_size]
                self._chunk_buf = self._chunk_buf[self._chunk_size:]

                # FireRedVAD expects int16
                i16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                for r in self.vad.detect_chunk(i16):
                    if r.is_speech_start:
                        self.in_speech = True
                    if r.is_speech_end:
                        self.in_speech = False
                        # Record segment end (start time tracked externally)
                        completed.append(("end", r.speech_end_frame / 100.0))
        elif self.backend == "silero":
            # Silero: feed in 512-sample windows
            window_size = 512
            for i in range(0, len(samples), window_size):
                chunk = samples[i:i + window_size]
                if len(chunk) < window_size:
                    chunk = np.pad(chunk, (0, window_size - len(chunk)))
                self.vad.accept_waveform(chunk)

                while not self.vad.empty():
                    seg = self.vad.front
                    self.vad.pop()
                    start = seg.start / 16000.0
                    if len(seg.samples) > 0:
                        end = start + len(seg.samples) / 16000.0
                    else:
                        end = start + 0.5  # estimate
                    completed.append((start, end))
        else:
            # Energy VAD
            completed = self.vad.accept_waveform(samples)

        return completed

    def is_speech(self) -> bool:
        """Check if currently in speech state."""
        if self.backend == "firered":
            return self.in_speech
        elif self.backend == "silero":
            return self.vad.is_speech_detected()
        else:
            return self.vad.is_speech()


class _EnergyStreamVad:
    """Simple energy-based streaming VAD for fallback."""

    def __init__(self, threshold=0.02, min_silence=0.5, min_speech=0.2, sr=16000):
        self.threshold = threshold
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.sr = sr
        self.in_speech = False
        self.speech_run = 0.0
        self.silence_run = 0.0
        self._start_time = 0.0
        self._total_samples = 0

    def reset(self):
        self.in_speech = False
        self.speech_run = 0.0
        self.silence_run = 0.0
        self._total_samples = 0

    def accept_waveform(self, window):
        window = np.asarray(window, dtype=np.float32)
        dt = len(window) / self.sr
        rms = float(np.sqrt(np.mean(window ** 2)) + 1e-9)
        completed = []

        if rms > self.threshold:
            self.speech_run += dt
            self.silence_run = 0.0
            if not self.in_speech and self.speech_run >= self.min_speech:
                self.in_speech = True
                self._start_time = self._total_samples / self.sr
        else:
            self.silence_run += dt
            self.speech_run = 0.0
            if self.in_speech and self.silence_run >= self.min_silence:
                self.in_speech = False
                end_time = (self._total_samples / self.sr) - self.silence_run
                completed.append((self._start_time, end_time))

        self._total_samples += len(window)
        return completed

    def is_speech(self):
        return self.in_speech


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    import soundfile as sf

    TEST_WAV = r"d:\HACKERMarathon\Project\dataset\Eval_Ali\Eval_Ali_near\audio_dir\R8001_M8004_N_SPK8013.wav"

    data, sr = sf.read(TEST_WAV, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    audio = data[:60 * sr]  # First 60 seconds

    print(f"Audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    print(f"FireRedVAD available: {os.path.isdir(_FIRERED_MODEL_DIR)}")
    print(f"Silero VAD available: {os.path.isfile(_SILERO_MODEL_PATH)}")

    # Test all three VADs
    for name, func in [
        ("FireRedVAD", firered_vad_segment),
        ("Silero VAD", silero_vad_segment),
        ("Energy VAD", energy_vad_segment),
    ]:
        print(f"\n--- {name} ---")
        result = func(audio, sr) if func else None
        if result is None:
            print("  Not available")
            continue
        durations = [e - s for s, e in result]
        print(f"  Segments: {len(result)}")
        if durations:
            print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s (avg {np.mean(durations):.1f}s)")
            for i, (s, e) in enumerate(result[:10]):
                print(f"    Seg {i+1}: [{s:.1f}s - {e:.1f}s] ({e-s:.1f}s)")

    # Test unified manager
    print("\n--- Unified VAD Manager ---")
    segments, vad_type = segment_audio(audio, sr)
    print(f"  Active VAD: {vad_type}")
    print(f"  Segments: {len(segments)}")
    durations = [e - s for s, e in segments]
    if durations:
        print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s (avg {np.mean(durations):.1f}s)")
