"""Persistent, validated hotword settings shared by the API and ASR runtime."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable


MIN_SCORE = 0.1
MAX_SCORE = 20.0
MAX_WORDS = 500
MAX_WORD_LENGTH = 80


def _contains_cjk(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" for char in value)


def _score(value, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return round(min(MAX_SCORE, max(MIN_SCORE, parsed)), 3)


class HotwordConfigStore:
    """Load and atomically persist the complete hotword configuration."""

    def __init__(self, path: str | Path, defaults: Iterable[str], default_score: float = 5.0):
        self.path = Path(path)
        self.defaults = [str(word).strip() for word in defaults if str(word).strip()]
        self.default_score = _score(default_score, 5.0)
        self._lock = threading.RLock()

    def load(self) -> dict:
        with self._lock:
            if self.path.is_file():
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                    return self.normalize(payload)
                except (OSError, ValueError, TypeError):
                    pass
            return self.normalize({
                "enabled": True,
                "fuzzy_pinyin_enabled": True,
                "default_score": self.default_score,
                "words": [{"text": word} for word in self.defaults],
            })

    def save(self, payload: dict) -> dict:
        normalized = self.normalize(payload)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        return normalized

    def add_words(self, words: Iterable[str]) -> dict:
        current = self.load()
        existing = list(current["words"])
        existing.extend({"text": word} for word in words)
        current["words"] = existing
        return self.save(current)

    def normalize(self, payload: dict | None) -> dict:
        source = payload if isinstance(payload, dict) else {}
        default_score = _score(source.get("default_score"), self.default_score)
        entries = source.get("words", [])
        if not isinstance(entries, list):
            entries = []

        words: list[dict] = []
        seen: set[str] = set()
        for raw in entries[:MAX_WORDS]:
            item = raw if isinstance(raw, dict) else {"text": raw}
            text = str(item.get("text", "")).strip()
            if not text or text.startswith("#"):
                continue
            text = text[:MAX_WORD_LENGTH].strip()
            key = text if _contains_cjk(text) else text.casefold()
            if key in seen:
                continue
            seen.add(key)
            inherited = default_score if _contains_cjk(text) else min(default_score, 2.5)
            words.append({
                "text": text,
                "score": _score(item.get("score"), inherited),
                "enabled": bool(item.get("enabled", True)),
            })

        return {
            "enabled": bool(source.get("enabled", True)),
            "fuzzy_pinyin_enabled": bool(source.get("fuzzy_pinyin_enabled", True)),
            "default_score": default_score,
            "words": words,
            "count": len(words),
            "active_count": sum(1 for word in words if word["enabled"]),
        }

    @staticmethod
    def engine_inputs(settings: dict) -> tuple[list[str], dict[str, float]]:
        if not settings.get("enabled", True):
            return [], {}
        active = [item for item in settings.get("words", []) if item.get("enabled", True)]
        words = [item["text"] for item in active]
        return words, {item["text"]: float(item["score"]) for item in active}
