"""Per-connection realtime audio session orchestration."""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    from modules.dtp2_protocol import RealtimePacket
    from modules.pcm_utils import float32_to_int16_pcm, safe_resample
    from modules.realtime_vad import RealtimeVAD
    from modules.rnnoise_filter import RNNoiseFilter
except Exception:  # pragma: no cover
    from dtp2_protocol import RealtimePacket
    from pcm_utils import float32_to_int16_pcm, safe_resample
    from realtime_vad import RealtimeVAD
    from rnnoise_filter import RNNoiseFilter


def format_live_result(
    result,
    *,
    display_text: Optional[str] = None,
    protocol: str = "json",
    seq: Optional[int] = None,
    denoise_report: Optional[dict[str, Any]] = None,
    vad_state: str = "speech",
    vad_endpoint: bool = False,
    rms: float = 0.0,
    asr_optimizer=None,
) -> dict[str, Any]:
    text = str(display_text if display_text is not None else (getattr(result, "text", "") or ""))
    is_final = bool(getattr(result, "is_final", False) or vad_endpoint)
    is_partial = bool(getattr(result, "is_partial", True) and not is_final)
    denoise = denoise_report or {"applied": False, "available": False}
    resp_data = {
        "timestamp": float(getattr(result, "timestamp", 0.0) or 0.0),
        "text": text,
        "raw_text": str(getattr(result, "raw_text", "") or text),
        "is_partial": is_partial,
        "is_final": is_final,
        "snr_db": float(getattr(result, "snr_db", 0.0) or 25.0),
        "quality_label": str(getattr(result, "quality_label", "medium") or "medium"),
        "asr_confidence": float(getattr(result, "asr_confidence", 0.0) or 0.8),
        "speaker_id": getattr(result, "speaker_id", "unknown"),
        "speaker_name": getattr(result, "speaker_name", None),
        "speaker_confidence": float(getattr(result, "speaker_confidence", 0.0) or 0.0),
        "overlap": bool(getattr(result, "overlap", False)),
        "overlap_speakers": list(getattr(result, "overlap_speakers", None) or []),
        "words": [
            {
                "text": getattr(word, "text", ""),
                "start_sec": float(getattr(word, "start_sec", 0.0) or 0.0),
                "end_sec": float(getattr(word, "end_sec", 0.0) or 0.0),
                "speaker_id": getattr(word, "speaker_id", "UNKNOWN"),
                "confidence": float(getattr(word, "confidence", 0.0) or 0.0),
            }
            for word in (getattr(result, "words", None) or [])
        ],
        "corrections": getattr(result, "corrections", None) or [],
        "logic_flags": getattr(result, "logic_flags", None) or [],
        "terms": getattr(result, "terms", None) or [],
        "uncertain_spans": getattr(result, "uncertain_spans", None) or [],
        "asr_optimizer": asr_optimizer,
        "protocol": protocol,
        "vad_state": vad_state,
        "vad_endpoint": bool(vad_endpoint),
        "rms": float(rms),
        "rnnoise_applied": bool(denoise.get("applied")),
        "rnnoise_available": bool(denoise.get("available")),
        "denoise": denoise,
        "postprocessed": bool(getattr(result, "postprocessed", False)),
        "fillers_removed": list(getattr(result, "fillers_removed", None) or []),
        "repetitions_merged": list(getattr(result, "repetitions_merged", None) or []),
    }
    if seq is not None:
        resp_data["seq"] = seq
    return resp_data


class EurekaLiveRealtimeSession:
    """Compatibility adapter for Eureka LiveAudioSession and HUIWU live JSON shape."""

    def __init__(self, live_session, engine=None, postprocess: Optional[Callable[[str], tuple[str, dict]]] = None, rnnoise_enabled: Optional[bool] = None):
        self.live_session = live_session
        self.engine = engine or getattr(live_session, "engine", None)
        self.postprocess = postprocess
        self.rnnoise = RNNoiseFilter(sample_rate=16000, enabled=rnnoise_enabled)
        self.chunk_count = 0
        self.send_count = 0
        self.protocol = "dtp2"
        self.closed = False

    def status(self) -> dict[str, Any]:
        metrics = self.live_session.metrics() if hasattr(self.live_session, "metrics") else {}
        return {
            "protocol": self.protocol,
            "chunks": self.chunk_count,
            "sent": self.send_count,
            "rnnoise": self.rnnoise.status(),
            "eureka": metrics,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if hasattr(self.live_session, "finish"):
                self.live_session.finish()
        finally:
            self.rnnoise.close()

    def set_denoise(self, enabled: Optional[bool]) -> None:
        if enabled is None:
            return
        current = bool(self.rnnoise.status().get("enabled"))
        if current == bool(enabled):
            return
        self.rnnoise.close()
        self.rnnoise = RNNoiseFilter(sample_rate=16000, enabled=bool(enabled))

    def handle_audio(self, packet: RealtimePacket) -> Optional[dict[str, Any]]:
        if self.closed or packet.audio is None:
            return None
        self.protocol = packet.protocol or self.protocol
        self.set_denoise(getattr(packet, "rnnoise_enabled", None))
        audio = safe_resample(packet.audio, packet.sample_rate, 16000)
        denoised, denoise_report = self.rnnoise.process_chunk(audio, 16000)
        payload = float32_to_int16_pcm(denoised)
        if not payload:
            return None
        if packet.protocol == "dtp2" and packet.seq is not None:
            frame = b"DTP2" + int(packet.seq).to_bytes(4, "little", signed=False) + payload
            return self.handle_dtp2_frame(frame, denoise_report=denoise_report, seq=packet.seq)
        return self._handle_results(
            self.live_session.push_pcm_s16le(payload),
            protocol=packet.protocol or self.protocol,
            seq=packet.seq,
            denoise_report=denoise_report,
        )

    def handle_dtp2_frame(self, frame: bytes, denoise_report: Optional[dict[str, Any]] = None, seq: Optional[int] = None) -> Optional[dict[str, Any]]:
        if self.closed:
            return None
        self.protocol = "dtp2"
        if seq is None and frame.startswith(b"DTP2") and len(frame) >= 8:
            seq = int.from_bytes(frame[4:8], "little", signed=False)
        self.chunk_count += 1
        return self._handle_results(
            self.live_session.push_binary_frame(frame),
            protocol="dtp2",
            seq=seq,
            denoise_report=denoise_report or {"applied": False, "available": False},
        )

    def _handle_results(self, results, *, protocol: str, seq: Optional[int], denoise_report: dict[str, Any]) -> Optional[dict[str, Any]]:
        result = next((item for item in (results or []) if getattr(item, "text", "") or getattr(item, "is_final", False)), None)
        if result is None:
            return None
        display_text = str(getattr(result, "text", "") or "")
        postproc_info = None
        if getattr(result, "is_final", False) and display_text and self.postprocess:
            try:
                display_text, postproc_info = self.postprocess(display_text)
            except Exception:
                postproc_info = None
        resp_data = format_live_result(
            result,
            display_text=display_text,
            protocol=protocol,
            seq=seq,
            denoise_report=denoise_report,
            vad_state=getattr(getattr(self.live_session, "vad", None), "provider_name", "eureka"),
            vad_endpoint=bool(getattr(result, "is_final", False)),
            rms=0.0,
            asr_optimizer=self.engine.get_asr_optimizer_report() if hasattr(self.engine, "get_asr_optimizer_report") else None,
        )
        if postproc_info:
            resp_data["postprocessed"] = True
            resp_data["original_text"] = postproc_info.get("original_text", "")
            resp_data["fillers_removed"] = postproc_info.get("fillers_removed", [])
            resp_data["corrections"] = postproc_info.get("corrections", [])
        self.send_count += 1
        return resp_data


class LiveRealtimeSession:
    """Glue layer for protocol audio, optional RNNoise, VAD and XASR."""

    def __init__(self, engine, postprocess: Optional[Callable[[str], tuple[str, dict]]] = None, rnnoise_enabled: Optional[bool] = None):
        self.engine = engine
        self.postprocess = postprocess
        self.vad = RealtimeVAD(sample_rate=16000)
        self.rnnoise = RNNoiseFilter(sample_rate=16000, enabled=rnnoise_enabled)
        self.chunk_count = 0
        self.send_count = 0
        self.protocol = "json"
        self.closed = False
        if self.engine:
            self.engine.start_session()

    def status(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "chunks": self.chunk_count,
            "sent": self.send_count,
            "rnnoise": self.rnnoise.status(),
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.engine:
                self.engine.end_session()
        finally:
            self.rnnoise.close()

    def set_denoise(self, enabled: Optional[bool]) -> None:
        """Enable or disable RNNoise for this live connection."""
        if enabled is None:
            return
        current = bool(self.rnnoise.status().get("enabled"))
        if current == bool(enabled):
            return
        self.rnnoise.close()
        self.rnnoise = RNNoiseFilter(sample_rate=16000, enabled=bool(enabled))

    def handle_audio(self, packet: RealtimePacket) -> Optional[dict[str, Any]]:
        if self.closed or self.engine is None or packet.audio is None:
            return None
        self.protocol = packet.protocol or self.protocol
        self.set_denoise(getattr(packet, "rnnoise_enabled", None))
        audio = safe_resample(packet.audio, packet.sample_rate, 16000)
        self.chunk_count += 1

        denoised, denoise_report = self.rnnoise.process_chunk(audio, 16000)
        vad_state = self.vad.update(denoised, 16000)
        result = self.engine.process_chunk(denoised, 16000)

        if not (getattr(result, "text", "") or getattr(result, "is_final", False) or vad_state.endpoint):
            return None

        display_text = str(getattr(result, "text", "") or "")
        postproc_info = None
        if getattr(result, "is_final", False) and display_text and self.postprocess:
            try:
                display_text, postproc_info = self.postprocess(display_text)
            except Exception:
                postproc_info = None

        is_final = bool(getattr(result, "is_final", False) or vad_state.endpoint)
        resp_data = format_live_result(
            result,
            display_text=display_text,
            protocol=self.protocol,
            seq=packet.seq,
            denoise_report=denoise_report,
            vad_state=vad_state.state,
            vad_endpoint=is_final,
            rms=float(vad_state.rms),
            asr_optimizer=self.engine.get_asr_optimizer_report() if hasattr(self.engine, "get_asr_optimizer_report") else None,
        )
        if postproc_info:
            resp_data["postprocessed"] = True
            resp_data["original_text"] = postproc_info.get("original_text", "")
            resp_data["fillers_removed"] = postproc_info.get("fillers_removed", [])
            resp_data["corrections"] = postproc_info.get("corrections", [])
        self.send_count += 1
        return resp_data
