#!/usr/bin/env python3
"""
谛听 DiTing - Standalone Demo Runner
==========================================================================
Processes an audio file end-to-end using X-ASR with full logging,
saving results to JSON and a human-readable transcript.

Usage:
    python run_demo.py <audio_file.mp3>
    python run_demo.py "D:\\dickrun\\20260712153902-....mp3"

Output:
    logs/diting.log       - Full debug log
    demo_output/
      transcript.txt      - Human-readable transcript
      results.json        - Full structured results
      stats.txt           - Processing statistics
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add backend to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

import numpy as np
from utils.logger import init_logging, get_logger
from xasr.asr_engine import XASREngine, ASRResult


def _sanitize(val):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {str(k): _sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize(v) for v in val]
    return val

# ===========================================================================
# Config
# ===========================================================================

# Domain hotwords for meeting scenarios
MEETING_HOTWORDS = [
    # Common meeting terms
    "产品", "项目", "技术", "运营", "市场", "用户", "客户",
    "需求", "方案", "预算", "进度", "排期", "上线", "测试",
    "版本", "迭代", "优化", "体验", "转化", "留存", "增长",
    # Academic
    "学院", "学生", "老师", "课程", "考试", "成绩", "学分",
    "宿舍", "安全", "管理", "通知", "安排", "会议", "报告",
    # Tech
    "AI", "模型", "数据", "服务器", "接口", "前端", "后端",
    "数据库", "算法", "框架", "部署", "运维", "监控",
    # Finance/numbers
    "Q1", "Q2", "Q3", "Q4", "OKR", "KPI", "ROI",
]

# ===========================================================================
# Main processing function
# ===========================================================================

def process_audio_file(
    file_path: str,
    output_dir: str = None,
    hotwords: list = None,
) -> dict:
    """
    Process an audio file end-to-end and return results.

    Args:
        file_path: Path to audio file (mp3/wav/flac)
        output_dir: Output directory (default: backend/demo_output/)
        hotwords: Custom hotwords (default: MEETING_HOTWORDS)

    Returns:
        dict with keys: results, stats, transcript
    """
    logger = get_logger("demo_runner")
    logger.info("=" * 70)
    logger.info("  DiTing Demo Runner - End-to-End Audio Processing")
    logger.info("=" * 70)

    # Validate input
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    file_name = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Input:   {file_name} ({file_size_mb:.1f} MB)")
    logger.info(f"Hotwords: {len(hotwords or MEETING_HOTWORDS)} domain terms")

    # Setup output
    if output_dir is None:
        output_dir = os.path.join(BACKEND_DIR, "demo_output")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(file_name)[0]

    # ── Initialize engine ──────────────────────────────────────
    logger.info("Initializing X-ASR engine...")
    t_init = time.time()

    engine = XASREngine(
        hotwords=hotwords or MEETING_HOTWORDS,
        enable_logic_validation=True,
        enable_hotword_correction=True,
        enable_uncertainty=True,
        enable_endpoint_detection=True,
        provider="cpu",
        num_threads=4,
    )

    if not engine.is_model_available:
        logger.error("X-ASR model not available! Check backend/xasr/models/")
        return {"error": "model_not_available", "results": []}

    logger.info(f"Engine ready in {time.time() - t_init:.1f}s")

    # ── Collect results with callbacks ─────────────────────────
    all_results = []
    progress_milestones = set()

    def on_segment(result: ASRResult, idx: int, total: int):
        """Called for each recognized utterance."""
        all_results.append(result)

        # Progress logging at milestones
        pct = int(idx / max(1, total) * 100)
        milestone = (pct // 10) * 10
        if milestone not in progress_milestones and milestone >= 10:
            progress_milestones.add(milestone)
            logger.info(f"  Progress: {milestone}% ({idx} utterances, "
                       f"latest: '{result.text[:60]}...' if len(result.text) > 60 else result.text)")

        # Always log first 5 segments
        if idx <= 5:
            logger.info(f"  Segment {idx}: [{result.start_sec:.1f}s-{result.end_sec:.1f}s] "
                       f"SNR={result.snr_db}dB quality={result.quality_label}")
            logger.info(f"    Text: {result.text[:120]}")

    def on_progress(stage: str, fraction: float):
        """Called for stage progress."""
        if stage in ("loading", "vad", "done"):
            logger.info(f"  Stage: {stage} ({fraction*100:.0f}%)")

    # ── Process ────────────────────────────────────────────────
    logger.info(f"Starting processing of {file_name}...")
    t_process = time.time()

    results = engine.process_file(
        file_path,
        on_segment=on_segment,
        on_progress=on_progress,
    )

    elapsed = time.time() - t_process
    audio_dur = sum(r.end_sec - r.start_sec for r in results) if results else 0

    # ── Statistics ─────────────────────────────────────────────
    stats = {
        "file_name": file_name,
        "file_size_mb": round(file_size_mb, 2),
        "elapsed_seconds": round(elapsed, 1),
        "real_time_factor": round(elapsed / max(1, audio_dur), 3),
        "total_utterances": len(results),
        "total_text_chars": sum(len(r.text) for r in results),
        "avg_confidence": round(
            sum(r.asr_confidence for r in results) / max(1, len(results)), 3
        ),
        "quality_distribution": {
            "high": sum(1 for r in results if r.quality_label == "high"),
            "medium": sum(1 for r in results if r.quality_label == "medium"),
            "low": sum(1 for r in results if r.quality_label == "low"),
        },
        "avg_snr_db": round(
            sum(r.snr_db for r in results) / max(1, len(results)), 1
        ),
        "logic_flags": sum(len(r.logic_flags) for r in results),
        "corrections": sum(len(r.corrections) for r in results),
        "terms_found": len(set(
            term for r in results for term in r.terms
        )),
        "uncertain_spans": sum(len(r.uncertain_spans) for r in results),
    }

    # ── Save results ───────────────────────────────────────────
    # JSON (full structured data)
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    json_output = {
        "metadata": {
            "engine": "X-ASR (sherpa-onnx zipformer2)",
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": file_name,
            "stats": stats,
        },
        "utterances": [
            {
                "index": i + 1,
                "start_sec": round(r.start_sec, 2),
                "end_sec": round(r.end_sec, 2),
                "text": r.text,
                "raw_text": r.raw_text,
                "snr_db": r.snr_db,
                "rt60": r.rt60,
                "quality_score": r.quality_score,
                "quality_label": r.quality_label,
                "asr_confidence": r.asr_confidence,
                "terms": r.terms,
                "data_points": r.data_points,
                "corrections": r.corrections,
                "logic_flags": r.logic_flags,
                "uncertain_spans": r.uncertain_spans,
                "uncertainty": r.uncertainty,
            }
            for i, r in enumerate(results)
        ],
    }
    json_output = _sanitize(json_output)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved: {json_path}")

    # Human-readable transcript
    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"谛听 DiTing - Meeting Transcript\n")
        f.write(f"File: {file_name}\n")
        f.write(f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stats: {stats['total_utterances']} utterances, "
                f"avg confidence {stats['avg_confidence']:.1%}\n")
        f.write("=" * 70 + "\n\n")

        for i, r in enumerate(results):
            ts = f"[{r.start_sec:7.1f}s - {r.end_sec:7.1f}s]"
            quality = f"[{r.quality_label.upper():6s} SNR={r.snr_db:4.1f}dB]"
            f.write(f"\n--- Utterance {i+1} {ts} {quality} ---\n")
            f.write(f"{r.text}\n")

            if r.corrections:
                for c in r.corrections:
                    f.write(f"  [Correction] {c.get('original','')} -> {c.get('corrected','')} "
                           f"({c.get('method','')})\n")
            if r.logic_flags:
                for lf in r.logic_flags:
                    f.write(f"  [Logic] {lf.get('message','')}\n")
            if r.uncertain_spans:
                for us in r.uncertain_spans:
                    f.write(f"  [Uncertain] '{us.get('text','')}' "
                           f"confidence={us.get('confidence',0):.0%}\n")
    logger.info(f"Transcript saved: {txt_path}")

    # Stats file
    stats_path = os.path.join(output_dir, f"{base_name}_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("DiTing Demo Runner - Processing Statistics\n")
        f.write("=" * 50 + "\n")
        for key, val in stats.items():
            f.write(f"  {key}: {val}\n")
    logger.info(f"Stats saved: {stats_path}")

    # ── Print summary ──────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  PROCESSING COMPLETE")
    logger.info(f"  Utterances:      {stats['total_utterances']}")
    logger.info(f"  Audio processed: {audio_dur:.0f}s in {elapsed:.1f}s (RTF: {stats['real_time_factor']})")
    logger.info(f"  Avg confidence:  {stats['avg_confidence']:.1%}")
    logger.info(f"  Quality:         High={stats['quality_distribution']['high']} "
               f"Med={stats['quality_distribution']['medium']} "
               f"Low={stats['quality_distribution']['low']}")
    logger.info(f"  Logic flags:     {stats['logic_flags']}")
    logger.info(f"  Corrections:     {stats['corrections']}")
    logger.info(f"  Output:          {output_dir}")
    logger.info("=" * 70)

    # Print first 10 utterances to console (handle Windows GBK safely)
    if results:
        print("\n" + "=" * 70)
        print("  FIRST 10 UTTERANCES:")
        print("=" * 70)
        for i, r in enumerate(results[:10]):
            info = f"\n  [{i+1}] {r.start_sec:.1f}s-{r.end_sec:.1f}s | SNR={r.snr_db}dB | {r.quality_label}"
            try:
                print(info)
            except UnicodeEncodeError:
                print(info.encode('ascii', errors='replace').decode('ascii'))
            try:
                print(f"  {r.text[:150]}")
            except UnicodeEncodeError:
                print(f"  (text has {len(r.text)} Chinese characters - see transcript file)")

    return {
        "results": results,
        "stats": stats,
        "output_dir": output_dir,
        "transcript_path": txt_path,
        "json_path": json_path,
    }


# ===========================================================================
# CLI Entrypoint
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DiTing Demo Runner - Process audio with X-ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py meeting.mp3
  python run_demo.py meeting.wav --hotwords "BERT,Transformer,Q3"
  python run_demo.py meeting.mp3 --output ./my_results/
        """
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--hotwords", "-w", default=None,
                       help="Comma-separated custom hotwords")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Initialize logging
    init_logging(console_level=args.log_level, file_level="DEBUG")

    # Parse hotwords
    hotwords = MEETING_HOTWORDS
    if args.hotwords:
        custom = [w.strip() for w in args.hotwords.split(",") if w.strip()]
        hotwords = list(set(MEETING_HOTWORDS + custom))

    # Process
    try:
        result = process_audio_file(
            file_path=args.audio_file,
            output_dir=args.output,
            hotwords=hotwords,
        )

        if result.get("error"):
            print(f"\nERROR: {result['error']}")
            sys.exit(1)

        print(f"\nDone! Transcript saved to: {result['transcript_path']}")
        print(f"Full results saved to: {result['json_path']}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
