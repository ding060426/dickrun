"""
谛听 会悟 - Centralized Logging System
==========================================================================
Provides structured logging to console, file, and a ring buffer for
WebSocket streaming. Supports log levels and module-scoped loggers.

Usage:
    from utils.logger import get_logger
    logger = get_logger("my_module")
    logger.info("Processing started")
    logger.warning("Low confidence segment detected")
"""

import logging
import logging.handlers
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ===========================================================================
# Ring buffer for WebSocket log streaming
# ===========================================================================

class RingBuffer:
    """Thread-safe ring buffer for recent log messages."""
    def __init__(self, max_entries: int = 500):
        self._buffer: List[Dict] = []
        self._max = max_entries
        self._lock = threading.Lock()

    def append(self, entry: Dict):
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > self._max:
                self._buffer = self._buffer[-self._max:]

    def get_recent(self, n: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._buffer[-n:])

    def clear(self):
        with self._lock:
            self._buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self._buffer)


# Global ring buffer instance
log_buffer = RingBuffer(max_entries=500)


# ===========================================================================
# Ring-buffer-aware handler
# ===========================================================================

class RingBufferHandler(logging.Handler):
    """Logging handler that also writes to the ring buffer."""
    def emit(self, record):
        try:
            entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'name': record.name,
                'message': self.format(record),
                'funcName': record.funcName,
                'lineno': record.lineno,
            }
            log_buffer.append(entry)
        except Exception:
            pass


# ===========================================================================
# Initialization
# ===========================================================================

_initialized = False
_LOG_DIR = None


def init_logging(
    log_dir: str = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    max_file_mb: int = 20,
    backup_count: int = 5,
):
    """
    Initialize the centralized logging system.

    Args:
        log_dir: Directory for log files (default: backend/logs/)
        console_level: Console log level
        file_level: File log level
        max_file_mb: Max log file size before rotation
        backup_count: Number of rotated backups to keep
    """
    global _initialized, _LOG_DIR

    if _initialized:
        return

    if log_dir is None:
        # Default to backend/logs/
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    _LOG_DIR = log_dir

    os.makedirs(log_dir, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers
    root.handlers.clear()

    # ── Console handler ──────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)-16s] %(message)s',
        datefmt='%H:%M:%S',
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # ── File handler ─────────────────────────────────────────────
    log_file = os.path.join(log_dir, "diting.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8',
    )
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)-16s] %(filename)s:%(lineno)d %(funcName)s() | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # ── Ring buffer handler ──────────────────────────────────────
    ring_handler = RingBufferHandler()
    ring_handler.setLevel(logging.DEBUG)
    ring_fmt = logging.Formatter('%(message)s')
    ring_handler.setFormatter(ring_fmt)
    root.addHandler(ring_handler)

    # ── Error file handler ───────────────────────────────────────
    error_file = os.path.join(log_dir, "errors.log")
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_fmt)
    root.addHandler(error_handler)

    _initialized = True

    # Log startup banner
    logger = logging.getLogger("diting")
    logger.info("=" * 60)
    logger.info("  会悟 Logging System Initialized")
    logger.info(f"  Log directory: {log_dir}")
    logger.info(f"  Console level: {console_level} | File level: {file_level}")
    logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger."""
    if not _initialized:
        init_logging()
    return logging.getLogger(name)


def get_recent_logs(n: int = 100) -> List[Dict]:
    """Get recent log entries from the ring buffer."""
    return log_buffer.get_recent(n)


def get_log_dir() -> Optional[str]:
    """Get the current log directory."""
    return _LOG_DIR
