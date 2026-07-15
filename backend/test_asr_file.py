#!/usr/bin/env python3
"""Validate VAD and X-ASR on a local real audio file.

Usage:
    python backend/test_asr_file.py path/to/audio.wav
    python backend/test_asr_file.py path/to/audio.wav --no-asr
    python backend/test_asr_file.py path/to/audio.wav --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.vad_manager import segment_audio, summarize_segments, get_vad_info
from xasr.asr_engine import XASREngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VAD and X-ASR on one local audio file.")
    parser.add_argument("audio_path", help="Path to wav/flac/mp3 audio file")
    parser.add_argument("--no-asr", action="store_true", help="Only run VAD segmentation; skip ASR decoding")
    parser.add_argument("--json", action="store_true", help="Print JSON report instead of human-readable output")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum VAD segment duration in seconds")
    parser.add_argument("--min-segment-duration", type=float, default=0.5, help="Minimum VAD segment duration in seconds")
    parser.add_argument("--pre-padding-ms", type=float, default=200.0, help="VAD pre-padding in milliseconds")
    parser.add_argument("--post-padding-ms", type=float, default=200.0, help="VAD post-padding in milliseconds")
    return parser.parse_args()


def build_engine() -> XASREngine:
    return XASREngine(
        enable_logic_validation=False,
        enable_hotword_correction=False,
        enable_uncertainty=False,
        enable_endpoint_detection=True,
        enable_cognitive=False,
        provider="cpu",
        num_threads=2,
    )


def run_vad_check(engine: XASREngine, audio_path: Path, args: argparse.Namespace) -> dict:
    data, sr = engine._load_audio(str(audio_path))
    duration = len(data) / sr if sr else 0.0
    segments, vad_type = segment_audio(
        data,
        sr,
        max_duration=args.max_duration,
        min_segment_duration=args.min_segment_duration,
        pre_padding_ms=args.pre_padding_ms,
        post_padding_ms=args.post_padding_ms,
    )
    summary = summarize_segments(segments, duration)
    return {
        "audio": {
            "path": str(audio_path),
            "duration_sec": round(duration, 3),
            "sample_rate": sr,
            "samples": int(len(data)),
        },
        "vad": {
            "backend": vad_type,
            "info": get_vad_info(),
            **summary,
            "segments": [
                {"start": round(s, 3), "end": round(e, 3), "duration": round(max(0.0, e - s), 3)}
                for s, e in segments
            ],
        },
    }


def run_asr_check(engine: XASREngine, audio_path: Path) -> dict:
    if not engine.is_model_available:
        return {
            "enabled": True,
            "model_available": False,
            "error": f"model files are missing: {engine.model_dir}",
        }

    progress_events = []

    def on_progress(stage: str, fraction: float) -> None:
        progress_events.append({"stage": stage, "fraction": round(float(fraction), 4)})

    t0 = time.time()
    results = engine.process_file(str(audio_path), on_progress=on_progress)
    elapsed = time.time() - t0

    non_empty = [r for r in results if (r.text or r.raw_text or "").strip()]
    total_chars = sum(len((r.text or r.raw_text or "").strip()) for r in results)
    avg_conf = sum(float(getattr(r, "asr_confidence", 0.0) or 0.0) for r in results) / len(results) if results else 0.0

    duration = 0.0
    if results:
        duration = max(float(getattr(r, "end_sec", 0.0) or 0.0) for r in results)

    return {
        "enabled": True,
        "model_available": True,
        "recognized_segments": len(results),
        "non_empty_text_segments": len(non_empty),
        "empty_text_segments": max(0, len(results) - len(non_empty)),
        "total_text_chars": total_chars,
        "avg_confidence": round(avg_conf, 4),
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration, 4) if duration > 0 else None,
        "progress_events": progress_events,
        "segments": [
            {
                "index": i,
                "start": round(float(r.start_sec), 3),
                "end": round(float(r.end_sec), 3),
                "duration": round(max(0.0, float(r.end_sec) - float(r.start_sec)), 3),
                "confidence": round(float(getattr(r, "asr_confidence", 0.0) or 0.0), 4),
                "snr_db": round(float(getattr(r, "snr_db", 0.0) or 0.0), 2),
                "quality": getattr(r, "quality_label", ""),
                "text": r.text or r.raw_text or "",
            }
            for i, r in enumerate(results, 1)
        ],
    }


def print_human_report(report: dict) -> None:
    audio = report["audio"]
    vad = report["vad"]
    print(f"[audio] {audio['path']}")
    print(f"[audio] duration={audio['duration_sec']:.1f}s sample_rate={audio['sample_rate']} samples={audio['samples']}")
    print(
        f"[vad] backend={vad['backend']} segments={vad['segments_count']} "
        f"coverage={vad['speech_coverage_ratio']:.2%}"
    )
    print(
        f"[vad] avg={vad['avg_segment_duration']:.2f}s "
        f"min={vad['min_segment_duration']:.2f}s max={vad['max_segment_duration']:.2f}s"
    )
    if vad.get("is_full_audio_fallback"):
        print("[vad] warning: likely full-audio fallback (single segment covers almost the whole file)")
    for seg in vad["segments"][:10]:
        print(f"  - {seg['start']:.2f}s -> {seg['end']:.2f}s ({seg['duration']:.2f}s)")
    if len(vad["segments"]) > 10:
        print(f"  ... {len(vad['segments']) - 10} more segments")

    asr = report.get("asr")
    if not asr:
        print("[asr] skipped")
        return
    if not asr.get("model_available", True):
        print(f"[asr] ERROR: {asr.get('error')}")
        return
    print(
        f"[asr] segments={asr['recognized_segments']} non_empty={asr['non_empty_text_segments']} "
        f"chars={asr['total_text_chars']} avg_conf={asr['avg_confidence']:.2f} "
        f"elapsed={asr['elapsed_seconds']:.1f}s rtf={asr['rtf']}"
    )
    for seg in asr["segments"][:20]:
        print(f"\n[{seg['index']}] {seg['start']:.2f}s - {seg['end']:.2f}s conf={seg['confidence']:.2f} snr={seg['snr_db']:.1f}")
        print(seg["text"])
    if len(asr["segments"]) > 20:
        print(f"\n... {len(asr['segments']) - 20} more ASR segments")


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    engine = build_engine()
    report = run_vad_check(engine, audio_path, args)
    if args.no_asr:
        report["asr"] = None
    else:
        report["asr"] = run_asr_check(engine, audio_path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human_report(report)

    vad = report["vad"]
    if vad["segments_count"] <= 0:
        return 1
    if not args.no_asr and report.get("asr", {}).get("model_available") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
