#!/usr/bin/env python3
"""
DiTing - One-click startup script v2.0
Starts backend API (with X-ASR engine) and frontend page.

Usage:
    python start.py              # Start all services
    python start.py --demo-only  # Skip X-ASR, demo mode only
"""

import os
import json
import sys
import subprocess
import time
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path

from backend.build_info import API_REVISION

ROOT_DIR = Path(__file__).parent.absolute()
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"
XASR_MODELS_DIR = BACKEND_DIR / "xasr" / "models"
BACKEND_HOST = os.environ.get("DITING_BACKEND_HOST", "127.0.0.1")
FRONTEND_HOST = os.environ.get("DITING_FRONTEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("DITING_BACKEND_PORT", "8765"))
FRONTEND_PORT = int(os.environ.get("DITING_FRONTEND_PORT", "3000"))
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}/?apiPort={BACKEND_PORT}"


def is_compatible_backend(health_info):
    """Return true only for the backend contract this frontend expects."""

    if not str(health_info.get("service", "")).startswith("DiTing"):
        return False
    try:
        return int(health_info.get("api_revision", -1)) >= API_REVISION
    except (TypeError, ValueError):
        return False


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
        [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", BACKEND_HOST, "--port", str(BACKEND_PORT),
        ],
        env=env,
    )

    # Wait for server to be ready
    for i in range(60):
        if proc.poll() is not None:
            raise RuntimeError(
                f"Backend exited with code {proc.returncode}; "
                f"port {BACKEND_PORT} may already be in use"
            )
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"{BACKEND_URL}/api/health")
            data = resp.read().decode()
            health_info = json.loads(data)
            if not is_compatible_backend(health_info):
                if proc.poll() is None:
                    proc.terminate()
                raise RuntimeError(
                    f"Port {BACKEND_PORT} is occupied by an old or incompatible "
                    "DiTing backend. Stop the earlier process, then start again."
                )
            print(f"[DiTing] Backend ready -> {BACKEND_URL}")
            print(f"[DiTing] API Docs  -> {BACKEND_URL}/docs")
            print(f"[DiTing] Logs     -> backend/logs/diting.log")

            # Check X-ASR status
            try:
                status = urllib.request.urlopen(f"{BACKEND_URL}/api/xasr/status")
                xasr_info = json.loads(status.read().decode())
                if xasr_info.get("available") and xasr_info.get("model_available"):
                    print(f"[DiTing] X-ASR Engine: READY ({xasr_info.get('model_dir', '')})")
                    print(f"[DiTing] Features: endpoint_detection={xasr_info.get('endpoint_detection', False)}")
                else:
                    print(f"[DiTing] X-ASR Engine: Demo mode (loading={xasr_info.get('loading', False)})")
            except Exception:
                pass
            return proc
        except RuntimeError:
            raise
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

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer((FRONTEND_HOST, FRONTEND_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[DiTing] Frontend ready -> {FRONTEND_URL}")
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
    print(f"  Backend API:   {BACKEND_URL}")
    print(f"  API Docs:      {BACKEND_URL}/docs")
    print(f"  X-ASR Status:  {BACKEND_URL}/api/xasr/status")
    print(f"  Eval Status:   {BACKEND_URL}/api/eval/status")
    print(f"  Demo Page:     {FRONTEND_URL}")
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
        webbrowser.open(FRONTEND_URL)
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
