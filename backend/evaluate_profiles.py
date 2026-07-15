#!/usr/bin/env python3
"""A/B compare deployed X-ASR profiles on a small labelled local corpus."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval_ali_integration import _compute_cer
from xasr.asr_engine import XASREngine
from xasr.config import ASR_CHUNK_PROFILES


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON array")
    items = []
    for raw in payload:
        if not isinstance(raw, dict) or not raw.get("audio") or raw.get("reference") is None:
            continue
        audio = Path(raw["audio"])
        if not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        items.append({
            "audio": audio,
            "reference": str(raw["reference"]),
            "keywords": [str(word) for word in raw.get("keywords", []) if str(word)],
        })
    return items


def evaluate_profiles(items: list[dict], profiles: list[str], model_dir: Path) -> dict:
    report = {"items": len(items), "profiles": {}}
    for profile in profiles:
        engine = XASREngine(
            model_dir=str(model_dir),
            asr_profile=profile,
            enable_hotword_correction=False,
            enable_logic_validation=False,
            enable_uncertainty=False,
        ).warmup()
        if not engine.is_model_available:
            report["profiles"][profile] = {"available": False}
            continue
        started = time.perf_counter()
        reference_all = ""
        hypothesis_all = ""
        keyword_hits = 0
        keyword_total = 0
        samples = []
        for item in items:
            results = engine.process_file(str(item["audio"]))
            hypothesis = "".join(result.raw_text or result.text for result in results)
            reference = item["reference"]
            reference_all += reference
            hypothesis_all += hypothesis
            hits = sum(1 for word in item["keywords"] if word.casefold() in hypothesis.casefold())
            keyword_hits += hits
            keyword_total += len(item["keywords"])
            samples.append({
                "audio": str(item["audio"]),
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": round(_compute_cer(reference, hypothesis), 4),
                "keyword_hits": hits,
                "keyword_total": len(item["keywords"]),
            })
        report["profiles"][profile] = {
            "available": True,
            "chunk_ms": ASR_CHUNK_PROFILES[profile],
            "cer": round(_compute_cer(reference_all, hypothesis_all), 4),
            "keyword_recall": round(keyword_hits / max(1, keyword_total), 4),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "samples": samples,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--profiles", nargs="+", choices=tuple(ASR_CHUNK_PROFILES), default=["low-latency", "meeting"])
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "xasr" / "models")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_profiles(load_manifest(args.manifest), args.profiles, args.model_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
