"""Bounded streaming storage for uploaded media files."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass


class UploadTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class StoredUpload:
    path: str
    size_bytes: int


async def save_upload_to_temp(
    upload,
    *,
    suffix: str,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> StoredUpload:
    """Copy an async upload stream without retaining the whole file in RAM."""
    if max_bytes <= 0 or chunk_size <= 0:
        raise ValueError("max_bytes and chunk_size must be positive")
    path = None
    size_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as target:
            path = target.name
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise UploadTooLargeError(
                        f"upload exceeds configured limit of {max_bytes} bytes"
                    )
                target.write(chunk)
        return StoredUpload(path=path, size_bytes=size_bytes)
    except Exception:
        if path and os.path.exists(path):
            os.unlink(path)
        raise
