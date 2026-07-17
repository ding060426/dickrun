"""In-memory upload job tracking and event fan-out."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"complete", "completed", "cancelled", "error"}


@dataclass
class UploadJob:
    file_id: str
    filename: str = ""
    source_path: Path | None = None
    status: str = "queued"
    progress: float = 0.0
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    events: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "source_path": str(self.source_path) if self.source_path else None,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cancelled": self.cancel_event.is_set(),
        }


class UploadJobStore:
    def __init__(self):
        self._jobs: dict[str, UploadJob] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = threading.RLock()

    def create(self, file_id: str, *, filename: str = "", source_path: str | Path | None = None) -> UploadJob:
        with self._lock:
            job = self._jobs.get(file_id)
            if job is None:
                job = UploadJob(file_id=file_id)
                self._jobs[file_id] = job
            job.filename = filename or job.filename
            job.source_path = Path(source_path) if source_path else job.source_path
            job.updated_at = time.time()
            return job

    def get(self, file_id: str) -> UploadJob | None:
        with self._lock:
            return self._jobs.get(file_id)

    def snapshot(self, file_id: str) -> dict | None:
        job = self.get(file_id)
        return job.snapshot() if job else None

    def counts(self) -> dict:
        with self._lock:
            queued = sum(1 for job in self._jobs.values() if job.status == "queued")
            running = sum(1 for job in self._jobs.values() if job.status not in TERMINAL_STATUSES and job.status != "queued")
            return {"queued": queued, "running": running, "total": len(self._jobs)}

    def update(self, file_id: str, *, status: str | None = None, progress: float | None = None, result: dict | None = None, error: str | None = None) -> UploadJob:
        with self._lock:
            job = self._jobs.setdefault(file_id, UploadJob(file_id=file_id))
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0.0, min(1.0, float(progress)))
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = time.time()
            return job

    def cancel(self, file_id: str) -> UploadJob | None:
        with self._lock:
            job = self._jobs.get(file_id)
            if job is None:
                return None
            job.cancel_event.set()
            job.status = "cancelled" if job.status in {"queued"} else job.status
            job.updated_at = time.time()
            return job

    def subscribe(self, file_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(file_id, set()).add(queue)
            job = self._jobs.get(file_id)
            if job:
                for event in job.events[-20:]:
                    queue.put_nowait(event)
        return queue

    def unsubscribe(self, file_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(file_id)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(file_id, None)

    def publish(self, file_id: str, event: dict, loop=None) -> None:
        with self._lock:
            job = self._jobs.setdefault(file_id, UploadJob(file_id=file_id))
            job.events.append(event)
            job.updated_at = time.time()
            subscribers = list(self._subscribers.get(file_id, set()))
        for queue in subscribers:
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            else:
                queue.put_nowait(event)

    def cleanup_expired(self, ttl_sec: float = 3600) -> None:
        cutoff = time.time() - ttl_sec
        with self._lock:
            for file_id, job in list(self._jobs.items()):
                if job.updated_at < cutoff and job.status in TERMINAL_STATUSES:
                    self._jobs.pop(file_id, None)
                    self._subscribers.pop(file_id, None)
