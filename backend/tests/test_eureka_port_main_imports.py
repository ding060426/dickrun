from pathlib import Path
from types import SimpleNamespace
import wave

from xasr.contracts import ASRResult, WordTimestamp
from diarization import (
    ChunkedDiarizationBackend,
    DiarizationSegment,
    MeetingRegistry,
    OfflineMeetingPipeline,
    SherpaDiarizationBackend,
)


def test_diarization_package_exports_integration_interfaces():
    assert OfflineMeetingPipeline is not None
    assert SherpaDiarizationBackend is not None
    assert ChunkedDiarizationBackend is not None
    assert MeetingRegistry is not None


def test_diarization_backend_fails_open_when_models_missing(tmp_path):
    backend = SherpaDiarizationBackend(
        tmp_path / "missing-segmentation.onnx",
        tmp_path / "missing-embedding.onnx",
        num_threads=1,
    )
    available, reason = backend.availability()
    assert available is False
    assert "missing" in reason or "sherpa-onnx" in reason


def test_offline_pipeline_exposes_status(tmp_path):
    backend = SherpaDiarizationBackend(
        tmp_path / "missing-segmentation.onnx",
        tmp_path / "missing-embedding.onnx",
        num_threads=1,
    )
    pipeline = OfflineMeetingPipeline(backend)
    status = pipeline.status()
    assert status["available"] is False
    assert status["provider"] == "sherpa-pyannote-3dspeaker"


class _RecordingEngine:
    def __init__(self):
        self.calls = []

    def process_file(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return [ASRResult(text="recognized")]


class _AvailableDiarizationBackend:
    provider_name = "test-diarization"

    def availability(self):
        return True, "available"

    def diarize(self, audio, sample_rate, num_speakers=None, on_progress=None):
        return [DiarizationSegment(0.0, 1.0, "SPEAKER_00", confidence=0.9)]


class _FailingDiarizationBackend(_AvailableDiarizationBackend):
    def diarize(self, audio, sample_rate, num_speakers=None, on_progress=None):
        raise RuntimeError("boom")


def _write_silent_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)


def test_offline_pipeline_success_forwards_cancel_and_denoise_to_asr_engine(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path)
    cancel_event = object()
    engine = _RecordingEngine()

    pipeline = OfflineMeetingPipeline(_AvailableDiarizationBackend())
    run = pipeline.process_file(
        audio_path,
        engine,
        cancel_event=cancel_event,
        denoise_enabled=True,
    )

    assert run.applied is True
    assert len(engine.calls) == 1
    assert engine.calls[0][1]["cancel_event"] is cancel_event
    assert engine.calls[0][1]["denoise_enabled"] is True


def test_offline_pipeline_runtime_fallback_forwards_cancel_and_denoise_to_asr_engine(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path)
    cancel_event = object()
    engine = _RecordingEngine()

    pipeline = OfflineMeetingPipeline(_FailingDiarizationBackend())
    run = pipeline.process_file(
        audio_path,
        engine,
        cancel_event=cancel_event,
        denoise_enabled=True,
    )

    assert run.applied is False
    assert "runtime error" in run.reason
    assert len(engine.calls) == 1
    assert engine.calls[0][1]["cancel_event"] is cancel_event
    assert engine.calls[0][1]["denoise_enabled"] is True


def test_main_imports_eureka_startup_interfaces_fail_open():
    import main

    assert hasattr(main, "HAS_EUREKA_PIPELINE")
    assert hasattr(main, "final_xasr_engine")
    assert hasattr(main, "xasr_pool")
    assert hasattr(main, "hotword_config_store")
    assert hasattr(main, "runtime_config_store")
    assert hasattr(main, "meeting_pipeline")
    assert hasattr(main, "meeting_registry")
    assert hasattr(main, "PROCESSING_EXECUTOR")

    assert main.meeting_pipeline is None or hasattr(main.meeting_pipeline, "status")
    assert main.meeting_registry is None or isinstance(main.meeting_registry, MeetingRegistry)


def test_main_build_meeting_pipeline_fails_open_when_models_missing(monkeypatch, tmp_path):
    import main

    monkeypatch.setattr(main, "BACKEND_DIR", str(tmp_path))
    pipeline = main._build_meeting_pipeline()
    assert pipeline is None or pipeline.status()["available"] is False


def test_result_to_dict_includes_eureka_diarization_and_postprocess_fields():
    import main

    result = ASRResult(
        text="clean text",
        raw_text="raw text",
        start_sec=1.25,
        end_sec=2.5,
        speaker_id="SPEAKER_01",
        speaker_name="Alice",
        speaker_confidence=0.87,
        overlap=True,
        overlap_speakers=["SPEAKER_01", "SPEAKER_02"],
        words=[WordTimestamp("clean", 1.25, 1.75, speaker_id="SPEAKER_01", confidence=0.91)],
        postprocessed=True,
        original_text="um clean text",
        fillers_removed=["um"],
        repetitions_merged=["clean clean"],
    )

    data = main._result_to_dict(result, 1)

    assert data["speaker_id"] == "SPEAKER_01"
    assert data["speaker_name"] == "Alice"
    assert data["speaker_confidence"] == 0.87
    assert data["overlap"] is True
    assert data["overlap_speakers"] == ["SPEAKER_01", "SPEAKER_02"]
    assert data["words"] == [
        {
            "text": "clean",
            "start_sec": 1.25,
            "end_sec": 1.75,
            "speaker_id": "SPEAKER_01",
            "confidence": 0.91,
        }
    ]
    assert data["postprocessed"] is True
    assert data["original_text"] == "um clean text"
    assert data["fillers_removed"] == ["um"]
    assert data["repetitions_merged"] == ["clean clean"]


def test_upload_helper_uses_final_engine_and_meeting_pipeline(monkeypatch, tmp_path):
    import main

    class FinalEngine:
        is_model_available = True

        def __init__(self):
            self.calls = []

        def process_file(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return [ASRResult(text="fallback")]

    class Pipeline:
        def __init__(self):
            self.engine = None

        def process_file(self, path, engine, **kwargs):
            self.engine = engine
            assert kwargs["enable_diarization"] is True
            return SimpleNamespace(
                results=[ASRResult(text="diarized", speaker_id="SPEAKER_00")],
                metadata=lambda: {"enabled": True, "applied": True, "provider": "test-pipeline"},
            )

    final_engine = FinalEngine()
    fallback_live_engine = FinalEngine()
    pipeline = Pipeline()
    monkeypatch.setattr(main, "final_xasr_engine", final_engine)
    monkeypatch.setattr(main, "xasr_engine", fallback_live_engine)
    monkeypatch.setattr(main, "meeting_pipeline", pipeline)

    results, metadata = main._process_uploaded_file_with_eureka_pipeline(str(tmp_path / "audio.wav"))

    assert [r.text for r in results] == ["diarized"]
    assert metadata == {"enabled": True, "applied": True, "provider": "test-pipeline"}
    assert pipeline.engine is final_engine
    assert final_engine.calls == []
    assert fallback_live_engine.calls == []


def test_upload_helper_falls_back_to_asr_only_when_pipeline_fails(monkeypatch, tmp_path):
    import main

    class Engine:
        is_model_available = True

        def __init__(self):
            self.kwargs = None

        def process_file(self, path, **kwargs):
            self.kwargs = kwargs
            return [ASRResult(text="asr only")]

        def get_asr_optimizer_report(self):
            return {"engine": "final"}

    class FailingPipeline:
        def process_file(self, *args, **kwargs):
            raise RuntimeError("diarization unavailable")

    engine = Engine()
    monkeypatch.setattr(main, "final_xasr_engine", engine)
    monkeypatch.setattr(main, "xasr_engine", None)
    monkeypatch.setattr(main, "meeting_pipeline", FailingPipeline())

    results, metadata = main._process_uploaded_file_with_eureka_pipeline(
        str(tmp_path / "audio.wav"), denoise_enabled=True
    )

    assert [r.text for r in results] == ["asr only"]
    assert metadata == {"enabled": False, "applied": False, "provider": "asr-only"}
    assert engine.kwargs["denoise_enabled"] is True


def test_upload_asr_optimizer_uses_final_upload_engine(monkeypatch):
    import main

    class Engine:
        def __init__(self, name):
            self.name = name

        def get_asr_optimizer_report(self):
            return {"engine": self.name}

    monkeypatch.setattr(main, "final_xasr_engine", Engine("final"))
    monkeypatch.setattr(main, "xasr_engine", Engine("live"))

    assert main._get_upload_asr_engine().get_asr_optimizer_report() == {"engine": "final"}


def test_upload_logic_validator_reset_targets_upload_engine(monkeypatch):
    import main

    class Validator:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    final_validator = Validator()
    live_validator = Validator()
    monkeypatch.setattr(main, "final_xasr_engine", SimpleNamespace(logic_validator=final_validator))
    monkeypatch.setattr(main, "xasr_engine", SimpleNamespace(logic_validator=live_validator))

    main._reset_upload_logic_validator()

    assert final_validator.reset_calls == 1
    assert live_validator.reset_calls == 0


def test_upload_processing_uses_shared_executor(monkeypatch):
    import main

    class ExplodingThreadPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("per-upload executor should not be created")

    monkeypatch.setattr(main.concurrent.futures, "ThreadPoolExecutor", ExplodingThreadPoolExecutor)
    submitted = []
    monkeypatch.setattr(main.PROCESSING_EXECUTOR, "submit", lambda fn: submitted.append(fn) or SimpleNamespace())

    main._submit_upload_processing(lambda: None)

    assert len(submitted) == 1


def test_upload_audio_uses_bounded_storage_and_returns_413_on_too_large(monkeypatch):
    import asyncio
    import main

    class FakeUpload:
        filename = "too-large.wav"

        async def read(self, size=-1):
            raise AssertionError("upload_audio must use save_upload_to_temp, not unbounded file.read()")

    async def fake_save_upload_to_temp(upload, **kwargs):
        assert upload.filename == "too-large.wav"
        assert kwargs["max_bytes"] > 0
        raise main.UploadTooLargeError("too large")

    monkeypatch.setattr(main, "save_upload_to_temp", fake_save_upload_to_temp)

    response = asyncio.run(main.upload_audio(FakeUpload(), file_id="file-1"))

    assert response.status_code == 413


def test_live_asr_engine_forks_from_pool_engine_when_available(monkeypatch):
    import main

    class BaseEngine:
        is_model_available = True

        def __init__(self):
            self.calls = []

        def fork_session(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(is_model_available=True)

    base_engine = BaseEngine()
    monkeypatch.setattr(main, "xasr_engine", base_engine)

    live_engine = main._get_live_asr_engine()

    assert live_engine.is_model_available is True
    assert base_engine.calls == [{"enable_endpoint_detection": True, "num_threads": 1}]


def test_live_asr_engine_constructs_compat_engine_without_fork(monkeypatch):
    import main

    class BaseEngine:
        is_model_available = True

    class CompatEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.is_model_available = True

    monkeypatch.setattr(main, "xasr_engine", BaseEngine())
    monkeypatch.setattr(main, "XASREngine", CompatEngine)

    live_engine = main._get_live_asr_engine()

    assert live_engine.kwargs["enable_endpoint_detection"] is False
    assert live_engine.kwargs["num_threads"] == 1
    assert live_engine.kwargs["provider"] == "cpu"


def test_live_realtime_response_includes_eureka_fields(monkeypatch):
    from modules.dtp2_protocol import RealtimePacket
    from modules import realtime_session

    class VadState:
        state = "speech"
        endpoint = False
        rms = 0.1

    class Vad:
        def __init__(self, sample_rate):
            pass

        def update(self, audio, sample_rate):
            return VadState()

    class RNNoise:
        def __init__(self, sample_rate=16000, enabled=None):
            pass

        def process_chunk(self, audio, sample_rate):
            return audio, {"applied": False, "available": False}

        def status(self):
            return {"enabled": False, "available": False}

        def close(self):
            pass

    class Engine:
        def start_session(self):
            pass

        def end_session(self):
            pass

        def process_chunk(self, audio, sample_rate):
            return ASRResult(
                text="hello",
                raw_text="hello",
                speaker_id="SPEAKER_01",
                speaker_name="Alice",
                speaker_confidence=0.73,
                overlap=True,
                overlap_speakers=["SPEAKER_01", "SPEAKER_02"],
                words=[WordTimestamp("hello", 0.0, 0.5, speaker_id="SPEAKER_01", confidence=0.88)],
                postprocessed=True,
                fillers_removed=["um"],
                repetitions_merged=["hello hello"],
            )

    monkeypatch.setattr(realtime_session, "RealtimeVAD", Vad)
    monkeypatch.setattr(realtime_session, "RNNoiseFilter", RNNoise)

    session = realtime_session.LiveRealtimeSession(Engine())
    try:
        data = session.handle_audio(RealtimePacket(kind="audio", audio=[0.0] * 160, sample_rate=16000, protocol="dtp2"))
    finally:
        session.close()

    assert data["speaker_name"] == "Alice"
    assert data["speaker_confidence"] == 0.73
    assert data["overlap"] is True
    assert data["overlap_speakers"] == ["SPEAKER_01", "SPEAKER_02"]
    assert data["words"] == [
        {
            "text": "hello",
            "start_sec": 0.0,
            "end_sec": 0.5,
            "speaker_id": "SPEAKER_01",
            "confidence": 0.88,
        }
    ]
    assert data["postprocessed"] is True
    assert data["fillers_removed"] == ["um"]
    assert data["repetitions_merged"] == ["hello hello"]


def test_live_session_factory_prefers_eureka_live_audio_session(monkeypatch):
    import main

    class Engine:
        is_model_available = True

        def fork_session(self, **kwargs):
            return SimpleNamespace(is_model_available=True, fork_kwargs=kwargs)

    class FakeVad:
        provider_name = "fake-vad"

    class FakeLiveAudioSession:
        def __init__(self, engine, **kwargs):
            self.engine = engine
            self.kwargs = kwargs
            self.closed = False

        def finish(self):
            self.closed = True
            return []

    vad = FakeVad()
    monkeypatch.setattr(main, "xasr_engine", Engine())
    monkeypatch.setattr(main, "LiveAudioSession", FakeLiveAudioSession)
    monkeypatch.setattr(main, "get_live_audio_profile", lambda: SimpleNamespace(name="meeting", pre_roll_ms=11, endpoint_grace_ms=22, tail_pad_ms=33))
    monkeypatch.setattr(main, "create_live_vad", lambda profile=None: vad)

    session = main._create_eureka_live_session(main._get_live_asr_engine())

    assert isinstance(session.live_session, FakeLiveAudioSession)
    assert session.live_session.engine.fork_kwargs == {"enable_endpoint_detection": True, "num_threads": 1}
    assert session.live_session.kwargs["vad"] is vad
    assert session.live_session.kwargs["pre_roll_ms"] == 11
    assert session.live_session.kwargs["endpoint_grace_ms"] == 22
    assert session.live_session.kwargs["tail_pad_ms"] == 33


def test_live_session_factory_uses_energy_vad_when_create_live_vad_fails(monkeypatch):
    import main

    class Engine:
        is_model_available = True

        def fork_session(self, **kwargs):
            return SimpleNamespace(is_model_available=True)

    class FakeEnergyVad:
        provider_name = "energy-vad"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLiveAudioSession:
        def __init__(self, engine, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(main, "xasr_engine", Engine())
    monkeypatch.setattr(main, "LiveAudioSession", FakeLiveAudioSession)
    monkeypatch.setattr(main, "get_live_audio_profile", lambda: SimpleNamespace(name="meeting", pre_roll_ms=1, endpoint_grace_ms=2, tail_pad_ms=3, vad_threshold=0.4, vad_min_silence=0.6, vad_min_speech=0.2))
    monkeypatch.setattr(main, "create_live_vad", lambda profile=None: (_ for _ in ()).throw(RuntimeError("silero missing")))
    monkeypatch.setattr(main, "EnergyVad", FakeEnergyVad)

    session = main._create_eureka_live_session(main._get_live_asr_engine())

    assert isinstance(session.live_session.kwargs["vad"], FakeEnergyVad)
    assert session.live_session.kwargs["vad"].kwargs["threshold"] == 0.4
    assert session.live_session.kwargs["vad"].kwargs["min_silence_duration"] == 0.6
    assert session.live_session.kwargs["vad"].kwargs["min_speech_duration"] == 0.2


def test_eureka_live_realtime_session_accepts_dtp2_frame_with_fake_engine(monkeypatch):
    from modules.realtime_session import EurekaLiveRealtimeSession

    class FakeResult:
        text = "hello"
        raw_text = "hello"
        is_partial = True
        is_final = False
        timestamp = 0.0
        snr_db = 25.0
        quality_label = "medium"
        asr_confidence = 0.8
        speaker_id = "unknown"
        corrections = []
        logic_flags = []
        terms = []
        uncertain_spans = []

    class FakeEurekaSession:
        def __init__(self):
            self.frames = []
            self.finished = False

        def push_binary_frame(self, frame):
            self.frames.append(frame)
            return [FakeResult()]

        def push_pcm_s16le(self, frame):
            self.frames.append(frame)
            return [FakeResult()]

        def finish(self):
            self.finished = True
            return []

    class RNNoise:
        def __init__(self, sample_rate=16000, enabled=None):
            self.enabled = enabled

        def process_chunk(self, audio, sample_rate):
            return audio, {"applied": False, "available": False}

        def status(self):
            return {"enabled": bool(self.enabled), "available": True}

        def close(self):
            pass

    monkeypatch.setattr("modules.realtime_session.RNNoiseFilter", RNNoise)
    eureka = FakeEurekaSession()
    session = EurekaLiveRealtimeSession(eureka)
    try:
        data = session.handle_dtp2_frame(
            b"DTP2" + (1).to_bytes(4, "little") + b"\x01\x00" * 320,
            denoise_report={"applied": False, "available": True},
        )
    finally:
        session.close()

    assert eureka.frames[0].startswith(b"DTP2" + (1).to_bytes(4, "little"))
    assert data["text"] == "hello"
    assert data["protocol"] == "dtp2"
    assert data["seq"] == 1
    assert data["rnnoise_available"] is True


def test_live_result_formatter_preserves_compatibility_fields():
    import main

    result = ASRResult(
        text="hello",
        raw_text="raw hello",
        speaker_id="SPEAKER_01",
        speaker_name="Alice",
        speaker_confidence=0.9,
        overlap=True,
        overlap_speakers=["SPEAKER_01", "SPEAKER_02"],
        words=[WordTimestamp("hello", 0.0, 0.5, speaker_id="SPEAKER_01", confidence=0.8)],
        postprocessed=True,
        fillers_removed=["um"],
        repetitions_merged=[],
    )

    data = main._live_result_to_response_data(
        result,
        protocol="dtp2",
        seq=7,
        denoise_report={"applied": False, "available": True},
        vad_state="eureka",
        vad_endpoint=False,
        rms=0.0,
        asr_optimizer={"ok": True},
    )

    for key in ["timestamp", "text", "raw_text", "is_partial", "is_final", "snr_db", "quality_label", "asr_confidence", "speaker_id", "corrections", "logic_flags", "terms", "uncertain_spans", "protocol"]:
        assert key in data
    assert data["speaker_name"] == "Alice"
    assert data["speaker_confidence"] == 0.9
    assert data["overlap"] is True
    assert data["words"][0]["text"] == "hello"
    assert data["postprocessed"] is True
    assert data["fillers_removed"] == ["um"]
    assert data["seq"] == 7
