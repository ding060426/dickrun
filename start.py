#!/usr/bin/env python3
"""
会悟 - One-click startup script v2.0
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


def resolve_backend_python(root_dir=ROOT_DIR, environ=None, current_python=None):
    """Choose the interpreter used by the backend service."""

    environ = os.environ if environ is None else environ
    current_python = sys.executable if current_python is None else current_python
    configured = str(environ.get("DITING_BACKEND_PYTHON", "")).strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise RuntimeError(
                "DITING_BACKEND_PYTHON does not point to an existing file: "
                f"{configured_path}"
            )
        return str(configured_path)

    candidates = (
        Path(root_dir) / ".venv-qwen3" / "Scripts" / "python.exe",
        Path(root_dir) / ".venv-qwen3" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(current_python)


def is_compatible_backend(health_info):
    """Return true only for the backend contract this frontend expects."""

    if not str(health_info.get("service", "")).startswith("会悟"):
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
        print(f"[会悟] WARNING: X-ASR models incomplete")
        print(f"          Found: {existing if existing else '(none)'}")
        print(f"          Missing: {missing}")
        print(f"          Model dir: {XASR_MODELS_DIR}")
        print(f"          Running in Demo mode (preset demo data)")
        print()
    else:
        total_mb = sum((XASR_MODELS_DIR / f).stat().st_size for f in required) / (1024*1024)
        print(f"[会悟] OK: X-ASR models ready ({total_mb:.0f}MB)")
        print()


def start_backend():
    """Start FastAPI backend."""
    backend_python = resolve_backend_python()
    print(f"[backend] Python: {backend_python}")
    print("[会悟] Starting backend service (with X-ASR engine v2.0)...")
    os.chdir(BACKEND_DIR)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [
            backend_python, "-m", "uvicorn", "main:app",
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
                    "会悟 backend. Stop the earlier process, then start again."
                )
            print(f"[会悟] Backend ready -> {BACKEND_URL}")
            print(f"[会悟] API Docs  -> {BACKEND_URL}/docs")
            print(f"[会悟] Logs     -> backend/logs/diting.log")

            # Check X-ASR status
            try:
                status = urllib.request.urlopen(f"{BACKEND_URL}/api/xasr/status")
                xasr_info = json.loads(status.read().decode())
                if xasr_info.get("available") and xasr_info.get("model_available"):
                    print(f"[会悟] X-ASR Engine: READY ({xasr_info.get('model_dir', '')})")
                    print(f"[会悟] Features: endpoint_detection={xasr_info.get('endpoint_detection', False)}")
                else:
                    print(f"[会悟] X-ASR Engine: Demo mode (loading={xasr_info.get('loading', False)})")
            except Exception:
                pass
            return proc
        except RuntimeError:
            raise
        except Exception:
            time.sleep(1.0)

    print("[会悟] WARNING: Backend start timeout (60s)")
    return proc


def start_frontend_server():
    """Start frontend static file server."""
    print("[会悟] Starting frontend page service...")
    os.chdir(FRONTEND_DIR)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer((FRONTEND_HOST, FRONTEND_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[会悟] Frontend ready -> {FRONTEND_URL}")
    return server


def main():
    print("=" * 62)
    print("  会悟 v2.0 - Smart Meeting Speech Cognitive System")
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
    print("  会悟 system is ready!")
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
        print("\n[会悟] Stopping services...")
        backend_proc.terminate()
        frontend_server.shutdown()
        print("[会悟] All services stopped.")


if __name__ == "__main__":
    main()
