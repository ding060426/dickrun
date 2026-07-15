"""
X-ASR Engine - DiTing Smart Meeting Speech Cognitive System
==========================================================================
Wraps sherpa-onnx streaming ASR with cognitive enhancement pipeline.

Pipeline:
  Audio -> VAD -> X-ASR(streaming zipformer2) -> Hotword Correction
       -> Logic Validation -> Uncertainty Estimation -> Enhanced Results

Key fix (2026-07-15): Added energy-based VAD pre-segmentation so that
process_file() produces per-utterance results instead of one giant blob.
Each speech segment gets its own sherpa-onnx stream.
"""

import os
import sys
import time
import logging
from typing import Optional, List, Dict, Callable
import numpy as np

from .contracts import ASRResult
from .config import resolve_asr_profile
from .hotwords import prepare_hotword_assets
from .sherpa_streaming_infer import (
    SherpaRecognizerRuntime,
    SherpaStreamingASR,
    format_text,
)

# Import cognitive enhancement modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.audio_processor import (
    estimate_snr, estimate_rt60, acoustic_quality_score,
    HotwordCorrector, LogicValidator, UncertaintyEstimator
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("xasr_engine")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ===========================================================================
# Simple energy-based VAD
# ===========================================================================

def _energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 25,
    hop_ms: int = 10,
    energy_threshold_ratio: float = 0.06,  # raised from 0.03 → filter background noise
    min_speech_frames: int = 15,           # ~150ms min speech (was 25=250ms)
    min_silence_frames: int = 30,          # ~300ms gap → sentence break (was 50=500ms)
    pre_padding_ms: float = 200,           # reduced from 300ms
    post_padding_ms: float = 200,          # reduced from 300ms
    min_segment_duration: float = 0.8,     # allow shorter independent segments (was 2.0s)
) -> List[tuple]:
    """
    Detect speech segments using energy-based VAD.

    Returns list of (start_sec, end_sec) tuples.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)

    if len(audio) < frame_len:
        return [(0.0, len(audio) / sample_rate)]

    # Compute short-time energy
    n_frames = (len(audio) - frame_len) // hop_len + 1
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        energies[i] = np.sqrt(np.mean(frame ** 2))

    # Dynamic threshold based on energy distribution
    if np.max(energies) > 0:
        energies_norm = energies / np.max(energies)
    else:
        return [(0.0, len(audio) / sample_rate)]

    threshold = max(energy_threshold_ratio, 0.5 * np.median(energies_norm))

    # Find speech frames
    is_speech = energies_norm > threshold

    # Merge close speech regions
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            if silence_count < min_silence_frames and segments:
                # Continue previous segment (short gap)
                segments[-1] = (segments[-1][0], 0)
                in_speech = True
                silence_count = 0
            else:
                in_speech = True
                speech_start = i
                silence_count = 0
        elif speech and in_speech:
            silence_count = 0
        elif not speech and in_speech:
            silence_count += 1
            if silence_count >= min_silence_frames:
                in_speech = False
                speech_end = i - silence_count
                duration_frames = speech_end - speech_start
                if duration_frames >= min_speech_frames:
                    start_sec = max(0, (speech_start * hop_len - pre_padding_ms / 1000 * sample_rate) / sample_rate)
                    end_sec = min(len(audio) / sample_rate,
                                  (speech_end * hop_len + post_padding_ms / 1000 * sample_rate) / sample_rate)
                    if end_sec > start_sec + 0.1:  # at least 100ms
                        segments.append((start_sec, end_sec))

    # Handle trailing speech
    if in_speech:
        i = len(is_speech) - 1
        while i >= 0 and is_speech[i]:
            i -= 1
        duration_frames = len(is_speech) - speech_start
        if duration_frames >= min_speech_frames:
            start_sec = max(0, (speech_start * hop_len - pre_padding_ms / 1000 * sample_rate)) / sample_rate
            end_sec = len(audio) / sample_rate
            if end_sec > start_sec + 0.1:
                segments.append((start_sec, end_sec))

    # If no segments found, return the whole audio
    if not segments:
        return [(0.0, len(audio) / sample_rate)]

    # Merge overlapping segments
    merged = []
    for seg in sorted(segments):
        if merged and seg[0] <= merged[-1][1] + 1.0:  # Within 1 second: merge
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    # Phase 2: merge short segments with neighbors
    if min_segment_duration > 0 and merged:
        final = []
        i = 0
        while i < len(merged):
            dur = merged[i][1] - merged[i][0]
            if dur < min_segment_duration:
                # Try to merge with previous
                if final and (merged[i][1] - final[-1][1]) < 2.0:
                    final[-1] = (final[-1][0], merged[i][1])
                # Try to merge with next
                elif i + 1 < len(merged) and (merged[i+1][0] - merged[i][0]) < 3.0:
                    merged[i+1] = (merged[i][0], merged[i+1][1])
                else:
                    final.append(merged[i])
            else:
                final.append(merged[i])
            i += 1
        merged = final

    logger.info(f"VAD: detected {len(merged)} speech segments in {len(audio)/sample_rate:.1f}s audio")
    return merged


# ===========================================================================
# XASREngine
# ===========================================================================

class XASREngine:
    """
    X-ASR Engine wrapping sherpa-onnx streaming ASR with cognitive enhancement.

    Usage:
        engine = XASREngine(hotwords=["BERT", "Q3", "转化率"])
        results = engine.process_file("meeting.wav")  # or .mp3
    """

    DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

    def __init__(
        self,
        hotwords: List[str] = None,
        enable_logic_validation: bool = True,
        enable_hotword_correction: bool = True,
        enable_uncertainty: bool = True,
        enable_endpoint_detection: bool = True,
        enable_text_postprocess: bool = True,
        model_dir: str = None,
        provider: str = "cpu",
        sample_rate: int = 16000,
        num_threads: int = 2,
        decoding_method: str = "greedy_search",
        model_type: str = "zipformer2",
        text_format: str = "none",
        speaker_id: str = "default",
        recognizer_runtime: SherpaRecognizerRuntime | None = None,
        asr_profile: str | None = None,
        hotwords_score: float = 5.0,
    ):
        """
        Initialize X-ASR engine.

        Args:
            hotwords: Domain hotword list
            enable_logic_validation: Enable logic validation
            enable_hotword_correction: Enable hotword correction
            enable_uncertainty: Enable uncertainty estimation
            enable_endpoint_detection: Enable sherpa-onnx endpoint detection
            enable_text_postprocess: Enable post-ASR text cleaning pipeline
                (filler removal + punctuation restoration + sentence splitting)
            model_dir: Model directory
            provider: ONNX inference backend (cpu / cuda / coreml)
            sample_rate: Audio sample rate
            num_threads: Inference threads
            decoding_method: Decoding method
            model_type: Model type
            text_format: Text formatting mode
            speaker_id: Current speaker ID
        """
        self.enable_logic_validation = enable_logic_validation
        self.enable_hotword_correction = enable_hotword_correction
        self.enable_uncertainty = enable_uncertainty
        self.enable_endpoint_detection = enable_endpoint_detection
        self.enable_text_postprocess = enable_text_postprocess
        self.speaker_id = speaker_id
        self._sample_rate = sample_rate
        self._num_threads = num_threads
        self._provider = provider
        self._decoding_method = decoding_method
        self._model_type = model_type
        self._text_format = text_format
        self._hotwords = list(hotwords or [])
        self._hotwords_score = hotwords_score
        self._recognizer_runtime = recognizer_runtime
        self.asr_profile, self.chunk_ms = resolve_asr_profile(asr_profile)

        # Model paths
        model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.model_dir = model_dir
        tokens = os.path.join(model_dir, "tokens.txt")
        encoder = os.path.join(model_dir, f"encoder-{self.chunk_ms}ms.onnx")
        decoder = os.path.join(model_dir, f"decoder-{self.chunk_ms}ms.onnx")
        joiner = os.path.join(model_dir, f"joiner-{self.chunk_ms}ms.onnx")

        # Check model availability
        self.model_available = all(os.path.exists(f) for f in [tokens, encoder, decoder, joiner])

        if self.model_available:
            total_mb = sum(os.path.getsize(f) for f in [tokens, encoder, decoder, joiner]) / (1024 * 1024)
            logger.info(f"X-ASR models ready ({total_mb:.0f}MB): {model_dir}")
            self._tokens = tokens
            self._encoder = encoder
            self._decoder = decoder
            self._joiner = joiner
            self.asr = None  # Created per-session for thread safety
        else:
            missing = [f for f in [tokens, encoder, decoder, joiner] if not os.path.exists(f)]
            logger.warning(f"X-ASR models missing: {missing}")
            self.asr = None

        # Cognitive enhancement modules
        self.hotword_corrector = HotwordCorrector(self._hotwords) if enable_hotword_correction else None
        self.logic_validator = LogicValidator() if enable_logic_validation else None
        self.uncertainty_estimator = UncertaintyEstimator() if enable_uncertainty else None

        # Session state
        self._session_active = False
        self._current_snr = 25.0
        self._current_rt60 = 0.3
        self._total_segments_processed = 0

    # ------------------------------------------------------------------
    # ASR instance factory (thread-safe, one per stream)
    # ------------------------------------------------------------------

    def _create_asr(self) -> Optional[SherpaStreamingASR]:
        """Create a fresh stream session backed by the shared runtime."""
        if not self.model_available:
            return None
        return self._get_recognizer_runtime().create_session()

    def _get_recognizer_runtime(self) -> SherpaRecognizerRuntime:
        if self._recognizer_runtime is None:
            hotword_assets = prepare_hotword_assets(
                self.model_dir,
                self._hotwords,
                score=self._hotwords_score,
            )
            self._recognizer_runtime = SherpaRecognizerRuntime(
                tokens=self._tokens,
                encoder=self._encoder,
                decoder=self._decoder,
                joiner=self._joiner,
                provider=self._provider,
                sample_rate=self._sample_rate,
                feature_dim=80,
                num_threads=self._num_threads,
                decoding_method=(
                    hotword_assets.decoding_method
                    if hotword_assets.enabled else self._decoding_method
                ),
                model_type=self._model_type,
                enable_endpoint_detection=self.enable_endpoint_detection,
                text_format=self._text_format,
                hotwords_file=(
                    str(hotword_assets.hotwords_file) if hotword_assets.hotwords_file else ""
                ),
                hotwords_score=self._hotwords_score,
                modeling_unit=hotword_assets.modeling_unit,
                bpe_vocab=str(hotword_assets.bpe_vocab) if hotword_assets.bpe_vocab else "",
                max_active_paths=hotword_assets.max_active_paths,
            )
        return self._recognizer_runtime

    def warmup(self) -> "XASREngine":
        """Load the recognizer runtime once during application startup."""
        if self.model_available:
            self._get_recognizer_runtime()
        return self

    def fork_session(self, **overrides) -> "XASREngine":
        """Create isolated enhancement/session state over the shared model runtime."""
        options = {
            "hotwords": list(self._hotwords),
            "enable_logic_validation": self.enable_logic_validation,
            "enable_hotword_correction": self.enable_hotword_correction,
            "enable_uncertainty": self.enable_uncertainty,
            "enable_endpoint_detection": self.enable_endpoint_detection,
            "enable_text_postprocess": self.enable_text_postprocess,
            "model_dir": self.model_dir,
            "provider": self._provider,
            "sample_rate": self._sample_rate,
            "num_threads": self._num_threads,
            "decoding_method": self._decoding_method,
            "model_type": self._model_type,
            "text_format": self._text_format,
            "speaker_id": self.speaker_id,
            "recognizer_runtime": self._get_recognizer_runtime(),
            "asr_profile": self.asr_profile,
            "hotwords_score": self._hotwords_score,
        }
        runtime_affecting = {
            "hotwords",
            "enable_endpoint_detection",
            "model_dir",
            "provider",
            "sample_rate",
            "num_threads",
            "decoding_method",
            "model_type",
            "text_format",
            "asr_profile",
            "hotwords_score",
        }
        if runtime_affecting.intersection(overrides):
            options["recognizer_runtime"] = None
        options.update(overrides)
        return XASREngine(**options)

    # ------------------------------------------------------------------
    # File processing (THE KEY FIX)
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: str,
        on_segment: Callable[[ASRResult, int, int], None] = None,
        on_progress: Callable[[str, float], None] = None,
    ) -> List[ASRResult]:
        """
        Process an entire audio file.

        NEW APPROACH (2026-07-15):
        1. Load audio
        2. Energy-based VAD to find speech segments
        3. For each speech segment: create a fresh ASR stream, process,
           get final result
        4. Return list of per-utterance results

        This fixes the previous bug where process_chunk() always returned
        is_final=False and no results were ever captured.

        Args:
            file_path: Path to audio file (wav/flac/mp3, 16kHz mono recommended)
            on_segment: Optional callback for each recognized utterance
            on_progress: Optional callback for progress updates (stage, fraction)

        Returns:
            List of ASRResult, one per recognized utterance
        """
        logger.info(f"=== Processing audio file: {os.path.basename(file_path)} ===")

        t0 = time.time()

        # 1. Load audio
        if on_progress:
            on_progress("loading", 0.0)
        data, sr = self._load_audio(file_path)
        duration = len(data) / sr
        logger.info(f"Audio loaded: {duration:.1f}s @ {sr}Hz, {len(data)} samples")

        # 2. VAD segmentation
        if on_progress:
            on_progress("vad", 0.05)
        segments = _energy_vad(
            data, sr,
            energy_threshold_ratio=0.06,   # filter out background hum & light breathing
            min_speech_frames=15,          # ~150ms — catch short utterances, skip clicks
            min_silence_frames=30,         # ~300ms — cut at natural pauses
            pre_padding_ms=200,
            post_padding_ms=200,
            min_segment_duration=0.8,      # keep brief but meaningful phrases
        )
        total_segments = len(segments)
        logger.info(f"VAD found {total_segments} speech segments")

        # 3. Process each speech segment
        # File mode creates an isolated recognizer for every VAD slice below;
        # do not also allocate a full live recognizer for the outer session.
        self.start_session(create_recognizer=False)
        results = []

        for seg_idx, (start_s, end_s) in enumerate(segments):
            progress = 0.1 + 0.85 * (seg_idx / max(1, total_segments))
            if on_progress:
                on_progress("processing", progress)

            start_sample = max(0, int(start_s * sr))
            end_sample = min(len(data), int(end_s * sr))
            seg_audio = data[start_sample:end_sample]

            seg_dur = (end_sample - start_sample) / sr
            if seg_dur < 0.1:
                logger.debug(f"  Segment {seg_idx}: too short ({seg_dur:.2f}s), skipping")
                continue

            logger.debug(f"  Segment {seg_idx}: {start_s:.1f}s - {end_s:.1f}s ({seg_dur:.1f}s)")

            # Process this segment with a fresh ASR stream
            result = self._process_speech_segment(seg_audio, sr, start_s, end_s)

            if result and result.text.strip():
                results.append(result)
                if on_segment:
                    on_segment(result, len(results), total_segments)

        # 4. Done
        self.end_session()
        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0.0
        logger.info(
            f"=== Processing complete: {len(results)} utterances "
            f"in {elapsed:.1f}s (RTF: {rtf:.3f}) ==="
        )

        if on_progress:
            on_progress("done", 1.0)

        return results

    def _process_speech_segment(
        self, audio: np.ndarray, sr: int, start_s: float, end_s: float
    ) -> Optional[ASRResult]:
        """
        Process a single speech segment with a fresh ASR stream.

        Uses either endpoint detection (if enabled) or processes the whole
        segment as one utterance.
        """
        # Estimate acoustic environment from this segment
        snr_db = estimate_snr(audio, sr)
        rt60 = estimate_rt60(audio, sr)
        self._current_snr = snr_db
        self._current_rt60 = rt60

        seg_dur = len(audio) / sr

        if seg_dur < 0.1:
            return None

        # Create a fresh ASR instance for this segment
        asr = self._create_asr()
        if asr is None:
            return self._build_result(
                raw_text="[Model not available]",
                is_partial=False, is_final=True,
                start_sec=start_s, end_sec=end_s,
            )

        try:
            # Feed non-overlapping chunks to the streaming recognizer.
            # Re-sending overlapped audio makes transducer states see duplicate speech
            # and can severely corrupt recognition on uploaded files.
            chunk_size = int(0.2 * sr)  # 200ms chunks
            final_texts = []

            # A short trailing silence helps streaming models flush the final tokens.
            tail_silence = np.zeros(int(0.5 * sr), dtype=np.float32)
            audio_for_asr = np.concatenate([audio.astype(np.float32, copy=False), tail_silence])

            for i in range(0, len(audio_for_asr), chunk_size):
                chunk = audio_for_asr[i:i + chunk_size]
                if len(chunk) == 0:
                    break

                asr.accept_waveform(chunk, sr)
                asr.decode()

            # The file-upload path is already segmented by VAD. Do not reset on
            # internal endpoint detection here; keep each VAD segment as one stream.
            asr.input_finished()
            remaining = asr.get_final_result()
            if remaining and remaining.strip():
                final_texts.append(remaining.strip())

            # Combine all utterances
            if not final_texts:
                # No endpoints detected - try getting final result directly
                final = asr.get_final_result()
                if final.strip():
                    final_texts = [final.strip()]

            combined_text = " ".join(final_texts) if final_texts else ""

            if not combined_text.strip():
                return None

            result = self._build_result(
                raw_text=combined_text,
                is_partial=False, is_final=True,
                start_sec=start_s, end_sec=end_s,
            )
            result.snr_db = round(snr_db, 1)
            result.rt60 = round(rt60, 2)
            result.audio_data = audio  # Store raw float32 audio for waveform encoding

            return result

        finally:
            # Clean up the ASR instance
            del asr

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, file_path: str):
        """Load audio file, supporting multiple formats."""
        ext = os.path.splitext(file_path)[1].lower()

        # Native formats via soundfile
        if ext in ('.wav', '.flac', '.ogg'):
            import soundfile as sf
            data, sr = sf.read(file_path, dtype="float32", always_2d=True)
            data = data.mean(axis=1)
            logger.info(f"Loaded via soundfile: {file_path} -> {len(data)/sr:.1f}s @ {sr}Hz")
            if sr != self._sample_rate:
                data = self._resample(data, sr, self._sample_rate)
                sr = self._sample_rate
            return data, sr

        # MP3 / MP4 via pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            sr = audio.frame_rate
            data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels > 1:
                data = data.reshape(-1, audio.channels).mean(axis=1)
            else:
                data = data.reshape(-1)
            max_val = np.abs(data).max()
            if max_val > 0:
                data = data / max_val
            if sr != self._sample_rate:
                data = self._resample(data, sr, self._sample_rate)
                sr = self._sample_rate
            logger.info(f"Loaded via pydub: {file_path} -> {len(data)/sr:.1f}s @ {sr}Hz")
            return data, sr
        except Exception as e1:
            logger.debug(f"pydub failed: {e1}")

        # Fallback: librosa
        try:
            import librosa
            data, sr = librosa.load(file_path, sr=self._sample_rate, mono=True)
            logger.info(f"Loaded via librosa: {file_path} -> {len(data)/sr:.1f}s @ {sr}Hz")
            return data, sr
        except Exception as e2:
            raise RuntimeError(
                f"Cannot decode audio file {file_path}: pydub={e1}, librosa={e2}"
            )

    # ------------------------------------------------------------------
    # Streaming (for real-time mic input)
    # ------------------------------------------------------------------

    def start_session(self, create_recognizer: bool = True):
        """Start a new recognition session (for streaming mode)."""
        if create_recognizer and self.asr:
            self.asr.reset()
        elif create_recognizer:
            # File recognition creates a short-lived recognizer per VAD slice,
            # while microphone recognition needs one recognizer for the whole
            # live session.  Construct it lazily so concurrent WebSockets do
            # not share sherpa-onnx stream state.
            self.asr = self._create_asr()
        self._session_active = True
        self._total_segments_processed = 0
        if self.logic_validator:
            self.logic_validator.reset()
        self._current_snr = 25.0
        self._current_rt60 = 0.3
        logger.debug("Session started")

    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = None) -> ASRResult:
        """
        Process a single audio chunk (for real-time streaming).

        Returns partial results. Use end_session() + _finalize_results()
        for the final result.
        """
        if not self._session_active:
            self.start_session()

        sr = sample_rate or self._sample_rate
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)

        if sr != self._sample_rate:
            audio_chunk = self._resample(audio_chunk, sr, self._sample_rate)

        self._total_segments_processed += 1

        # Acoustic estimation is substantially more expensive than accepting
        # one streaming chunk. Sample immediately, then periodically.
        if self._total_segments_processed == 1 or self._total_segments_processed % 5 == 0:
            self._current_snr = estimate_snr(audio_chunk, sr)
            self._current_rt60 = estimate_rt60(audio_chunk, sr)

        # X-ASR inference
        if self.asr:
            self.asr.accept_waveform(audio_chunk, self._sample_rate)
            self.asr.decode()
            raw_text = self.asr.get_partial_result()
            is_endpoint = self.asr.is_endpoint()
        else:
            raw_text = ""
            is_endpoint = False

        if not raw_text and not is_endpoint:
            return ASRResult(
                text="",
                raw_text="",
                is_partial=True,
                is_final=False,
                timestamp=time.time(),
                snr_db=round(self._current_snr, 1),
                rt60=round(self._current_rt60, 2),
                quality_label="medium",
            )

        result = self._build_result(
            raw_text=raw_text,
            is_partial=True,
            is_final=is_endpoint,
        )

        # If endpoint detected, get final text and reset
        if is_endpoint and self.asr:
            final_text = self.asr.get_final_result()
            if final_text.strip():
                result.text = final_text.strip()
                result.raw_text = final_text.strip()
                result.is_partial = False
                result.is_final = True
            self.asr.reset()

        return result

    def end_session(self):
        """End current session."""
        if self._session_active:
            if self.asr:
                self.asr.input_finished()
                self.asr = None
            self._session_active = False
            logger.debug("Session ended")

    def _finalize_results(self) -> Optional[ASRResult]:
        """Backward-compatible finalization for the end of a live session."""
        return self.finalize_utterance(reset_stream=False)

    def finalize_utterance(
        self,
        reset_stream: bool = True,
        tail_pad_ms: int = 0,
    ) -> Optional[ASRResult]:
        """Finalize one VAD utterance and optionally prepare the next stream."""
        if not self.asr:
            return None

        if tail_pad_ms > 0:
            tail = np.zeros(
                round(self._sample_rate * tail_pad_ms / 1000),
                dtype=np.float32,
            )
            self.asr.accept_waveform(tail, self._sample_rate)
            self.asr.decode()

        raw_text = self.asr.get_final_result()
        result = None
        if raw_text.strip():
            result = self._build_result(
                raw_text=raw_text,
                is_partial=False,
                is_final=True,
            )
        if reset_stream:
            self.asr.reset()
        else:
            self._session_active = False
            self.asr = None
        return result

    # ------------------------------------------------------------------
    # Cognitive enhancement pipeline
    # ------------------------------------------------------------------

    def _build_result(
        self, raw_text: str, is_partial: bool, is_final: bool,
        start_sec: float = 0.0, end_sec: float = 0.0,
    ) -> ASRResult:
        """Build a complete ASRResult with cognitive enhancement."""

        # 1. Acoustic quality
        quality = acoustic_quality_score(self._current_snr, self._current_rt60)
        quality_label = "high" if quality > 0.7 else ("medium" if quality > 0.4 else "low")
        asr_confidence = max(0.4, quality * 0.9)

        # 2. Hotword correction
        corrections = []
        display_text = raw_text
        terms = []
        if self.hotword_corrector and raw_text:
            corrections = self.hotword_corrector.pinyin_correct(raw_text)
            display_text = raw_text
            for c in sorted(corrections, key=lambda x: x.get('position', 0), reverse=True):
                orig = c.get('original', '')
                corr = c.get('corrected', '')
                if orig in display_text:
                    display_text = display_text.replace(orig, corr, 1)
            for hw in self.hotword_corrector.hotwords:
                if hw in display_text:
                    terms.append(hw)

        # 2.5 Text post-processing (filler removal + punctuation + sentence splitting)
        # Only apply to final results — partial results are shown as-is for low latency
        postprocessed = False
        original_text = display_text
        fillers_removed = []
        repetitions_merged = []
        if self.enable_text_postprocess and is_final and display_text.strip():
            from modules.text_post_processor import process_asr_text_with_details
            processed_text, postprocess_info = process_asr_text_with_details(
                display_text,
                enable_filler_filter=True,
                enable_punctuation=True,
                enable_force_split=True,
                enable_normalize=True,
            )
            if processed_text.strip():
                display_text = processed_text
            fillers_removed = postprocess_info["fillers_removed"]
            repetitions_merged = postprocess_info["repetitions_merged"]
            corrections.extend(postprocess_info["corrections"])
            postprocessed = display_text != original_text or bool(
                fillers_removed or repetitions_merged or postprocess_info["corrections"]
            )

        # 3. Data points extraction
        data_points = self._extract_data_points(display_text)

        # 4. Logic validation
        logic_flags = []
        if self.logic_validator and display_text and is_final:
            logic_flags = self.logic_validator.add_statement(
                self.speaker_id, display_text, data_points, time.time(),
            )

        # 5. Uncertain spans
        uncertain_spans = []
        if self.enable_uncertainty and self._current_snr < 12:
            words = display_text.split()
            for i, word in enumerate(words):
                if len(word) <= 2 and self._current_snr < 8:
                    uncertain_spans.append({
                        'start': i, 'end': i + 1, 'text': word,
                        'confidence': max(0.4, self._current_snr / 20),
                        'candidates': [word],
                    })

        # 6. Uncertainty estimation
        uncertainty = {}
        if self.enable_uncertainty:
            uncertainty = self.uncertainty_estimator.estimate(
                snr_db=self._current_snr, rt60=self._current_rt60,
                overlap_ratio=0.0, asr_confidence=asr_confidence,
            )

        if is_partial:
            logic_flags = []

        return ASRResult(
            text=display_text,
            raw_text=raw_text,
            is_partial=is_partial, is_final=is_final,
            timestamp=time.time(),
            start_sec=start_sec, end_sec=end_sec,
            asr_confidence=asr_confidence,
            snr_db=round(self._current_snr, 1),
            rt60=round(self._current_rt60, 2),
            quality_score=round(quality, 2),
            quality_label=quality_label,
            corrections=corrections,
            postprocessed=postprocessed,
            original_text=original_text,
            fillers_removed=fillers_removed,
            repetitions_merged=repetitions_merged,
            logic_flags=logic_flags,
            uncertainty=uncertainty,
            terms=terms, data_points=data_points,
            uncertain_spans=uncertain_spans,
        )

    def _extract_data_points(self, text: str) -> List[dict]:
        """Extract data points (numbers + units) from text."""
        import re
        dps = []

        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
            dps.append({'value': m.group() + '%', 'type': 'percentage', 'position': m.start()})

        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(万|亿|元)', text):
            dps.append({'value': m.group(), 'type': 'amount', 'position': m.start()})

        for m in re.finditer(r'(?<!\d)(\d{2,})(?!\d|\.?\d*%)', text):
            val = int(m.group())
            if val > 10:
                dps.append({'value': str(val), 'type': 'number', 'position': m.start()})

        return dps

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Linear resampling."""
        if orig_sr == target_sr:
            return audio
        n = int(round(len(audio) * target_sr / orig_sr))
        xp = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return np.interp(xq, xp, audio).astype(np.float32)

    def add_hotwords(self, words: List[str]):
        """Add hotwords and rebuild the shared runtime for future sessions."""
        changed = False
        for word in words:
            normalized = str(word).strip()
            if normalized and normalized not in self._hotwords:
                self._hotwords.append(normalized)
                changed = True
            if normalized and self.hotword_corrector:
                self.hotword_corrector.hotwords.add(normalized)
        if changed:
            self._recognizer_runtime = None

    def set_speaker(self, speaker_id: str):
        """Switch speaker."""
        self.speaker_id = speaker_id

    @property
    def is_model_available(self) -> bool:
        return self.model_available
