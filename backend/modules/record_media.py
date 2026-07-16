"""Source audio persistence for meeting records.

Each record gets a subdirectory under MEDIA_ROOT where the original
upload or full microphone recording is stored.  Segment audio remains
in the SQLite BLOB column — this module only manages *source* audio.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
MEDIA_ROOT = Path(
    os.environ.get("DITING_RECORD_MEDIA_DIR", BACKEND_DIR / "data" / "record_media")
)


def record_media_dir(record_id: str) -> Path:
    """Return the directory for a record's media files."""
    return MEDIA_ROOT / record_id


def ensure_record_media_dir(record_id: str) -> Path:
    """Create and return the record media directory (idempotent)."""
    path = record_media_dir(record_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file (streamed, 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ingest_source_audio(
    record_id: str,
    source_path: str | Path,
    *,
    keep_source: bool = True,
) -> dict:
    """Copy or move *source_path* into the record's media directory.

    Returns a dict suitable for patching into ``meeting_records``:
        source_audio_path, source_audio_sha256
    """
    ensure_record_media_dir(record_id)
    src = Path(source_path)
    dest_ext = src.suffix.lower() or ".wav"
    dest = record_media_dir(record_id) / f"source{dest_ext}"
    if keep_source:
        shutil.copy2(src, dest)
    else:
        shutil.move(src, dest)
    return {
        "source_audio_path": str(dest),
        "source_audio_sha256": sha256_file(dest),
    }


def delete_record_media(record_id: str) -> None:
    """Remove all media files for a record (best-effort)."""
    path = record_media_dir(record_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
