#!/usr/bin/env python3
"""
DiTing - One-click startup script v2.0
Starts backend API (with X-ASR engine) and frontend page.

Usage:
    python start.py              # Start all services
    python start.py --demo-only  # Skip X-ASR, demo mode only
"""

import os
import sys
import subprocess
import time
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"
XASR_MODELS_DIR = BACKEND_DIR / "xasr" / "models"


def check_xasr_models():
    """Check if X-ASR models exist."""
    required = ["encoder-160ms.onnx", "decoder-160ms.onnx", "joiner-160ms.onnx", "tokens.txt"]
    existing = []
    missing = []
    for f in required:
        path = XASR_MODELS_DIR / f
        if path.exists():
            existing.append(f)
        else:
            missing.append(f)

    if missing:
        print(f"[DiTing] WARNING: X-ASR models incomplete")
        print(f"          Found: {existing if existing else '(none)'}")
        print(f"          Missing: {missing}")
        print(f"          Model dir: {XASR_MODELS_DIR}")
        print(f"          Running in Demo mode (preset demo data)")
        print()
    else:
        total_mb = sum((XASR_MODELS_DIR / f).stat().st_size for f in required) / (1024*1024)
        print(f"[DiTing] OK: X-ASR models ready ({total_mb:.0f}MB)")
        print()


def start_backend():
    """Start FastAPI backend."""
    print("[DiTing] Starting backend service (with X-ASR engine v2.0)...")
    os.chdir(BACKEND_DIR)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"],
        env=env,
    )

    # Wait for server to be ready
    for i in range(60):
        try:
            import urllib.request
            resp = urllib.request.urlopen("http://localhost:8765/api/health")
            data = resp.read().decode()
            print(f"[DiTing] Backend ready -> http://localhost:8765")
            print(f"[DiTing] API Docs  -> http://localhost:8765/docs")
            print(f"[DiTing] Logs     -> backend/logs/diting.log")

            # Check X-ASR status
            import json
            try:
                status = urllib.request.urlopen("http://localhost:8765/api/xasr/status")
                xasr_info = json.loads(status.read().decode())
                if xasr_info.get("available") and xasr_info.get("model_available"):
                    print(f"[DiTing] X-ASR Engine: READY ({xasr_info.get('model_dir', '')})")
                    print(f"[DiTing] Features: endpoint_detection={xasr_info.get('endpoint_detection', False)}")
                else:
                    print(f"[DiTing] X-ASR Engine: Demo mode (loading={xasr_info.get('loading', False)})")
            except Exception:
                pass
            return proc
        except Exception:
            time.sleep(1.0)

    print("[DiTing] WARNING: Backend start timeout (60s)")
    return proc


def start_frontend_server():
    """Start frontend static file server."""
    print("[DiTing] Starting frontend page service...")
    os.chdir(FRONTEND_DIR)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    # Allow port reuse so restarts don't fail with WinError 10048
    socketserver.TCPServer.allow_reuse_address = True

    server = socketserver.TCPServer(("", 3000), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[DiTing] Frontend ready -> http://localhost:3000")
    return server


def main():
    print("=" * 62)
    print("  DiTing v2.0 - Smart Meeting Speech Cognitive System")
    print("  Environment x Hotwords x Logic Validation")
    print("  ASR Engine: X-ASR (sherpa-onnx zipformer2)")
    print("=" * 62)
    print()

    # Check models
    check_xasr_models()

    # Start backend
    backend_proc = start_backend()

    # Start frontend
    frontend_server = start_frontend_server()

    print()
    print("=" * 62)
    print("  DiTing system is ready!")
    print()
    print("  Backend API:   http://localhost:8765")
    print("  API Docs:      http://localhost:8765/docs")
    print("  X-ASR Status:  http://localhost:8765/api/xasr/status")
    print("  Eval Status:   http://localhost:8765/api/eval/status")
    print("  Demo Page:     http://localhost:3000")
    print()
    print("  Features:")
    print("    [Demo Mode]     Pre-recorded simulated meeting")
    print("    [Mic Record]    Real-time X-ASR via WebSocket")
    print("    [File Upload]   Upload wav/mp3 for ASR with live progress")
    print("    [Eval Dataset]  AliMeeting evaluation endpoints")
    print()
    print("  Press Ctrl+C to stop all services")
    print("=" * 62)

    # Open browser
    try:
        webbrowser.open("http://localhost:3000")
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[DiTing] Stopping services...")
        backend_proc.terminate()
        frontend_server.shutdown()
        print("[DiTing] All services stopped.")


if __name__ == "__main__":
    main()
