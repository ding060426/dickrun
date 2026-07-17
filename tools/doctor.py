#!/usr/bin/env python3
"""Diagnose Huiwu local runtime dependencies, models, ports, and APIs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.model_paths import inspect_model_paths


DEPENDENCIES = ("numpy", "sherpa_onnx", "soundfile", "librosa", "torch", "qwen_asr")


def _check_dependency(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"name": name, "available": spec is not None}


def _check_port(host: str, port: int) -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        occupied = sock.connect_ex((host, port)) == 0
    return {"host": host, "port": port, "occupied": occupied}


def _fetch_json(url: str, timeout: float = 2.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"ok": True, "status": response.status, "data": json.loads(response.read().decode("utf-8"))}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


def collect_report(backend_url: str, frontend_host: str = "127.0.0.1") -> dict:
    models = inspect_model_paths()
    xasr_profiles = models["xasr_profiles"]
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "dependencies": [_check_dependency(name) for name in DEPENDENCIES],
        "models": models,
        "ports": [
            _check_port(frontend_host, 8765),
            _check_port(frontend_host, 3000),
        ],
        "apis": {
            "health": _fetch_json(f"{backend_url.rstrip('/')}/api/health"),
            "xasr_status": _fetch_json(f"{backend_url.rstrip('/')}/api/xasr/status"),
        },
        "summary": {
            "xasr_available_profiles": [
                profile for profile, status in xasr_profiles.items() if status["complete"]
            ],
            "vad_available": models["vad"]["available"],
            "qwen3_model_available": models["qwen3"]["available"],
            "diarization_available": models["diarization"]["available"],
        },
    }


def _line(status: str, label: str, detail: str = "") -> str:
    suffix = f": {detail}" if detail else ""
    return f"[{status}] {label}{suffix}"


def print_human(report: dict) -> None:
    python_info = report["python"]
    print(_line("OK", "Python", f"{python_info['executable']} ({python_info['version']})"))

    for dep in report["dependencies"]:
        print(_line("OK" if dep["available"] else "WARN", f"Dependency {dep['name']}"))

    models = report["models"]
    for profile, status in models["xasr_profiles"].items():
        marker = "OK" if status["complete"] else "WARN"
        detail = f"{status['chunk_ms']}ms"
        if status["missing"]:
            detail += f", missing {', '.join(status['missing'])}"
        print(_line(marker, f"X-ASR {profile}", detail))

    print(_line("OK" if models["vad"]["available"] else "WARN", "Silero VAD", models["vad_model_path"]))
    qwen_detail = models["qwen3_model_dir"]
    if models["qwen3"]["missing"]:
        qwen_detail += f", missing {', '.join(models['qwen3']['missing'])}"
    print(_line("OK" if models["qwen3"]["available"] else "WARN", "Qwen3 model", qwen_detail))
    diar_detail = models["diarization_model_dir"]
    if models["diarization"]["missing"]:
        diar_detail += f", missing {len(models['diarization']['missing'])} file(s); ASR-only fallback"
    print(_line("OK" if models["diarization"]["available"] else "WARN", "Diarization", diar_detail))

    for port in report["ports"]:
        print(_line("WARN" if port["occupied"] else "OK", f"Port {port['port']}", "occupied" if port["occupied"] else "free"))

    for name, api in report["apis"].items():
        print(_line("OK" if api["ok"] else "WARN", f"API {name}", str(api.get("status") or api.get("error", ""))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    report = collect_report(args.backend_url)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)


if __name__ == "__main__":
    main()
