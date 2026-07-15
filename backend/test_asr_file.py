#!/usr/bin/env python3
"""Run X-ASR on one local audio file and print per-segment results.

Usage:
    python backend/test_asr_file.py path/to/audio.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.asr_engine import XASREngine


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python backend/test_asr_file.py path/to/audio.wav")
        return 2

    audio_path = Path(sys.argv[1]).expanduser().resolve()
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}")
        return 1

    engine = XASREngine(
        enable_logic_validation=False,
        enable_hotword_correction=False,
        enable_uncertainty=False,
        enable_endpoint_detection=True,
        provider="cpu",
        num_threads=2,
    )

    if not engine.is_model_available:
        print(f"ERROR: model files are missing: {engine.model_dir}")
        return 1

    def on_progress(stage: str, fraction: float) -> None:
        print(f"[progress] {stage}: {fraction:.0%}")

    results = engine.process_file(str(audio_path), on_progress=on_progress)

    print(f"\nsegments: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.start_sec:.2f}s - {result.end_sec:.2f}s")
        print(result.text or result.raw_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
