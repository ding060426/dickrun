"""
X-ASR Engine - DiTing Smart Meeting Speech Cognitive System
==========================================================================
Wraps sherpa-onnx streaming ASR with cognitive enhancement pipeline.

Pipeline:
  Audio -> VAD -> X-ASR(streaming zipformer2) -> Hotword Correction
       -> Logic Validation -> Uncertainty Estimation -> Enhanced Results

File recognition uses the deployed local Silero VAD model so uploaded and
canonical recordings are segmented consistently without heuristic energy cuts.
"""

import os
import sys
import time
import logging
from typing import Optional, List, Dict, Callable
import numpy as np

from audio_buffer import AudioBuffer, load_audio_buffer
from .contracts import ASRResult
from .config import resolve_asr_profile
from .file_vad import SileroFileVad
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
    LogicValidator, UncertaintyEstimator
)
from modules.hotword_processor import HotwordProcessor

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
        hotword_scores: Dict[str, float] = None,
        enable_fuzzy_pinyin: bool = True,
        file_vad_options: Dict | None = None,
        file_segmenter=None,
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
        self._hotword_scores = dict(hotword_scores or {})
        self.enable_fuzzy_pinyin = enable_fuzzy_pinyin
        self._recognizer_runtime = recognizer_runtime
        self.asr_profile, self.chunk_ms = resolve_asr_profile(asr_profile)

        # Model paths
        model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.model_dir = model_dir
        self._file_vad_options = dict(file_vad_options or {})
        self.file_segmenter = file_segmenter or self._create_file_segmenter()
        self.file_vad_provider = (
            getattr(self.file_segmenter, "provider_name", type(self.file_segmenter).__name__)
            if self.file_segmenter is not None
            else "whole-file-fallback"
        )
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
        self.hotword_corrector = (
            HotwordProcessor(
                self._hotwords,
                fuzzy_pinyin_enabled=self.enable_fuzzy_pinyin,
            )
            if enable_hotword_correction and self._hotwords else None
        )
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
                scores=self._hotword_scores,
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
            "hotword_scores": dict(self._hotword_scores),
            "enable_fuzzy_pinyin": self.enable_fuzzy_pinyin,
            "file_vad_options": dict(self._file_vad_options),
            "file_segmenter": self.file_segmenter,
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
            "hotword_scores",
            "file_vad_options",
            "file_segmenter",
        }
        if runtime_affecting.intersection(overrides):
            options["recognizer_runtime"] = None
        options.update(overrides)
        return XASREngine(**options)

    def _create_file_segmenter(self):
        model_path = os.path.join(self.model_dir, "silero_vad.onnx")
        if not os.path.isfile(model_path):
            logger.warning("Silero file VAD model missing; file recognition will use whole audio")
            return None
        return SileroFileVad(model_path, **self._file_vad_options)

    # ------------------------------------------------------------------
    # File processing (THE KEY FIX)
    # ------------------------------------------------------------------

    def process_file(
        self,
        file_path: str,
        on_segment: Callable[[ASRResult, int, int], None] = None,
        on_progress: Callable[[str, float], None] = None,
        audio_buffer: AudioBuffer | None = None,
    ) -> List[ASRResult]:
        """
        Process an entire audio file.

        File recognition pipeline:
        1. Load audio
        2. Local Silero VAD finds speech segments
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
        if audio_buffer is None:
            data, sr = self._load_audio(file_path)
        else:
            data, sr = audio_buffer.samples, audio_buffer.sample_rate
        duration = len(data) / sr
        logger.info(f"Audio loaded: {duration:.1f}s @ {sr}Hz, {len(data)} samples")

        if len(data) == 0:
            if on_progress:
                on_progress("done", 1.0)
            return []

        # 2. VAD segmentation
        if on_progress:
            on_progress("vad", 0.05)
        segments = (
            self.file_segmenter.detect(data, sr)
            if self.file_segmenter is not None
            else [(0.0, duration)]
        )
        total_segments = len(segments)
        logger.info(
            "%s found %d speech segments",
            self.file_vad_provider,
            total_segments,
        )

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
            tail_silence = np.zeros(int(1.0 * sr), dtype=np.float32)
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

    def recognize_interval(
        self,
        audio_buffer: AudioBuffer,
        start_sec: float,
        end_sec: float,
        *,
        pre_padding_ms: int = 250,
        post_padding_ms: int = 400,
    ) -> Optional[ASRResult]:
        """Re-decode one speaker-change interval with protected boundaries."""

        start_sec = max(0.0, float(start_sec))
        end_sec = min(audio_buffer.duration, float(end_sec))
        if end_sec <= start_sec:
            return None
        decode_start = max(0.0, start_sec - max(0, pre_padding_ms) / 1000)
        decode_end = min(
            audio_buffer.duration,
            end_sec + max(0, post_padding_ms) / 1000,
        )
        result = self._process_speech_segment(
            audio_buffer.slice(decode_start, decode_end),
            audio_buffer.sample_rate,
            start_sec,
            end_sec,
        )
        if result is not None:
            result.audio_data = audio_buffer.slice(start_sec, end_sec)
        return result

    def _load_audio(self, file_path: str):
        """Compatibility wrapper around the canonical shared audio loader."""

        audio = load_audio_buffer(file_path, target_sample_rate=self._sample_rate)
        logger.info(
            "Loaded canonical audio: %s -> %.1fs @ %dHz mono float32",
            file_path,
            audio.duration,
            audio.sample_rate,
        )
        return audio.samples, audio.sample_rate

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
            display_text, corrections = self.hotword_corrector.rewrite(raw_text)
            terms = self.hotword_corrector.matched_terms(display_text)

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
        merged = list(self._hotwords)
        for word in words:
            normalized = str(word).strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        self.configure_hotwords(
            merged,
            scores=self._hotword_scores,
            default_score=self._hotwords_score,
            enabled=self.enable_hotword_correction,
            fuzzy_pinyin_enabled=self.enable_fuzzy_pinyin,
        )

    def configure_hotwords(
        self,
        words: List[str],
        *,
        scores: Dict[str, float] = None,
        default_score: float = None,
        enabled: bool = True,
        fuzzy_pinyin_enabled: bool = True,
    ):
        """Apply hotword settings to future streams without disrupting active ones."""
        self._hotwords = [str(word).strip() for word in words if str(word).strip()]
        self._hotword_scores = dict(scores or {})
        if default_score is not None:
            self._hotwords_score = float(default_score)
        self.enable_hotword_correction = bool(enabled)
        self.enable_fuzzy_pinyin = bool(fuzzy_pinyin_enabled)
        self.hotword_corrector = (
            HotwordProcessor(
                self._hotwords,
                fuzzy_pinyin_enabled=self.enable_fuzzy_pinyin,
            )
            if self.enable_hotword_correction and self._hotwords else None
        )
        self._recognizer_runtime = None

    def set_speaker(self, speaker_id: str):
        """Switch speaker."""
        self.speaker_id = speaker_id

    @property
    def is_model_available(self) -> bool:
        return self.model_available
