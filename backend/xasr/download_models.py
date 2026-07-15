#!/usr/bin/env python3
"""Download one local X-ASR streaming profile from the official HF repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import requests

try:
    from .config import ASR_CHUNK_PROFILES
except ImportError:  # direct script execution
    from config import ASR_CHUNK_PROFILES


MODEL_DIR = Path(__file__).parent / "models"
REPO_ID = "GilgameshWind/X-ASR-zh-en"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"


def profile_files(profile: str) -> list[tuple[str, str]]:
    chunk_ms = ASR_CHUNK_PROFILES[profile]
    remote_root = f"deployment/models/chunk-{chunk_ms}ms-model"
    names = [
        f"encoder-{chunk_ms}ms.onnx",
        f"decoder-{chunk_ms}ms.onnx",
        f"joiner-{chunk_ms}ms.onnx",
        "tokens.txt",
    ]
    return [(f"{remote_root}/{name}", name) for name in names]


def download_file(remote_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    existing = temporary.stat().st_size if temporary.is_file() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    url = f"{BASE_URL}/{quote(remote_path, safe='/')}?download=true"
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        if existing and response.status_code != 206:
            existing = 0
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0")) + existing
        mode = "ab" if existing else "wb"
        downloaded = existing
        last_percent = -1
        with temporary.open(mode) as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                output.write(block)
                downloaded += len(block)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent >= last_percent + 5:
                        print(f"    {percent:3d}%  {downloaded / 1024 / 1024:.1f} MB", flush=True)
                        last_percent = percent
    if temporary.stat().st_size < (1024 if destination.name == "tokens.txt" else 1024 * 1024):
        raise RuntimeError(f"Downloaded file is unexpectedly small: {temporary}")
    os.replace(temporary, destination)


def download_profile(profile: str, model_dir: Path = MODEL_DIR) -> None:
    chunk_ms = ASR_CHUNK_PROFILES[profile]
    print(f"X-ASR {profile} ({chunk_ms} ms) -> {model_dir}")
    for remote_path, filename in profile_files(profile):
        destination = model_dir / filename
        if destination.is_file() and destination.stat().st_size >= (
            1024 if filename == "tokens.txt" else 1024 * 1024
        ):
            print(f"  skip {filename} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        print(f"  download {filename}")
        download_file(remote_path, destination)
        print(f"  ready {filename} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=tuple(ASR_CHUNK_PROFILES),
        default="meeting",
        help="Local ASR profile to deploy (default: meeting / 960 ms)",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    download_profile(args.profile, args.model_dir)


if __name__ == "__main__":
    main()
