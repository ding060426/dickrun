from xasr.contracts import ASRResult, WordTimestamp
from xasr.config import resolve_asr_profile


def test_asr_result_contains_eureka_fields():
    result = ASRResult(text="hello")
    assert result.text == "hello"
    assert result.speaker_id == "UNKNOWN"
    assert result.speaker_name is None
    assert result.speaker_confidence == 0.0
    assert result.overlap is False
    assert result.overlap_speakers == []
    assert result.words == []
    assert result.postprocessed is False
    assert result.fillers_removed == []
    assert result.repetitions_merged == []


def test_word_timestamp_contract():
    word = WordTimestamp(text="你好", start_sec=0.1, end_sec=0.4, speaker_id="SPEAKER_00", confidence=0.9)
    assert word.text == "你好"
    assert word.start_sec == 0.1
    assert word.end_sec == 0.4
    assert word.speaker_id == "SPEAKER_00"
    assert word.confidence == 0.9


def test_asr_profile_resolution():
    assert resolve_asr_profile("low-latency") == ("low-latency", 160)
    assert resolve_asr_profile("balanced") == ("balanced", 480)
    assert resolve_asr_profile("meeting") == ("meeting", 960)
    assert resolve_asr_profile("quality") == ("quality", 1920)


def test_integrated_asr_engine_imports_with_missing_model():
    from xasr.asr_engine import ASRResult as EngineASRResult, XASREngine

    engine = XASREngine(hotwords=["BERT"], model_dir="missing-model-dir")

    assert EngineASRResult is ASRResult
    assert engine.is_model_available is False
    assert hasattr(engine, "process_file")
    assert hasattr(engine, "process_chunk")
    assert hasattr(engine, "fork_session")
    assert isinstance(engine.get_asr_optimizer_report(), dict)
