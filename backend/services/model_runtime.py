"""Model runtime orchestration for live and final ASR engines."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from xasr.engine_pool import AsrEnginePool


class ModelRuntimeService:
    """Own model pool loading, reload coalescing, and lightweight runtime access."""

    def __init__(
        self,
        *,
        has_xasr: bool,
        model_dir: str | Path,
        runtime_config_store,
        hotword_config_store,
        logger,
        base_options: dict | None = None,
        pool_factory: Callable[..., Any] = AsrEnginePool,
    ):
        self.has_xasr = bool(has_xasr)
        self.model_dir = Path(model_dir)
        self.runtime_config_store = runtime_config_store
        self.hotword_config_store = hotword_config_store
        self.logger = logger
        self.base_options = dict(base_options or {})
        self.pool_factory = pool_factory
        self.pool = None
        self.live_engine = None
        self.final_engine = None
        self.loading = False
        self._reload_lock = threading.Lock()
        self._reload_pending = False
        self._reload_worker_active = False

    def engine_label(self, engine) -> str:
        return str(getattr(engine, "engine_name", "X-ASR (sherpa-onnx zipformer2 v2.0)"))

    def load(self) -> dict:
        """Build and publish configured live/final ASR runtimes."""
        if not self.has_xasr:
            self.logger.info("X-ASR not available; transcription is disabled")
            return self.status()
        self.loading = True
        self.logger.info("Loading configured X-ASR live/final runtimes...")
        try:
            runtime_settings = self.runtime_config_store.load()
            hotword_settings = self.hotword_config_store.load()
            pool = self.pool
            if not isinstance(pool, AsrEnginePool):
                pool = self.pool_factory(self.model_dir, base_options=self.base_options)
            status = pool.reload(runtime_settings["recognition"], hotword_settings)
            self.pool = pool
            self.live_engine = pool.live_engine
            self.final_engine = pool.final_engine
            self.logger.info(
                "X-ASR ready: live=%sms final=%sms shared=%s file_vad=%s threads=%s",
                status["live"].get("chunk_ms"),
                status["final"].get("chunk_ms"),
                status.get("shared_runtime"),
                status.get("file_vad_provider"),
                status.get("inference_threads"),
            )
            return status
        except Exception as error:
            self.logger.error("X-ASR init failed: %s", error)
            self.logger.error(traceback.format_exc())
            return self.status()
        finally:
            self.loading = False

    def _reload_worker(self) -> None:
        while True:
            with self._reload_lock:
                if not self._reload_pending:
                    self._reload_worker_active = False
                    return
                self._reload_pending = False
            self.load()

    def schedule_reload(self, on_complete: Callable[[], None] | None = None) -> None:
        with self._reload_lock:
            self._reload_pending = True
            if self._reload_worker_active:
                return
            self._reload_worker_active = True
        def run() -> None:
            self._reload_worker()
            if on_complete is not None:
                on_complete()
        threading.Thread(target=run, daemon=True).start()

    def configure_hotwords(self, settings: dict) -> None:
        if self.pool:
            self.pool.configure_hotwords(settings)

    def create_live_session(self):
        return self.pool.create_live_session() if self.pool else None

    def create_final_session(self):
        return self.pool.create_final_session() if self.pool else None

    def status(self) -> dict:
        return self.pool.status() if self.pool else {}

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
        self.pool = None
        self.live_engine = None
        self.final_engine = None
        self.loading = False
