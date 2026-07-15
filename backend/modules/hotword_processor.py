"""X-ASR-style hotword canonicalization and fuzzy-pinyin correction."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

try:
    from pypinyin import Style, pinyin
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps ASR usable
    Style = None
    pinyin = None


_CJK = re.compile(r"^[\u3400-\u9fff\uf900-\ufaff]+$")
_LEGACY_ALIASES = {
    "BERT": ["bat", "bate", "bot"],
    "Transformer": ["chuansi", "chuan si former"],
    "A/B": ["ab"],
    "A/B Test": ["ab test", "a b test"],
    "Q3": ["q 3"],
    "OKR": ["o k r"],
}


def fuzzy_pinyin_key(value: str) -> str:
    """Collapse common Mandarin accent confusions into a comparison key."""
    key = str(value or "").strip().lower().replace("ü", "v").replace("u:", "v")
    if key.startswith(("zh", "ch", "sh")):
        key = key[0] + key[2:]
    if key.startswith("n"):
        key = "l" + key[1:]
    if key.endswith("ing"):
        key = key[:-3] + "in"
    elif key.endswith("eng"):
        key = key[:-3] + "en"
    elif key.endswith("ang"):
        key = key[:-3] + "an"
    return key


@lru_cache(maxsize=8192)
def _character_readings(character: str) -> frozenset[str]:
    if pinyin is None or not _CJK.fullmatch(character):
        return frozenset()
    values = pinyin(character, style=Style.NORMAL, heteronym=True, strict=False)
    readings = values[0] if values else []
    return frozenset(fuzzy_pinyin_key(value) for value in readings if value)


def _ascii_aliases(canonical: str) -> set[str]:
    aliases = {canonical}
    if not canonical.isascii():
        return aliases
    compact = canonical.replace(" ", "")
    aliases.add(compact)
    if "." in compact:
        aliases.update({compact.replace(".", ""), compact.replace(".", " ")})
    if "-" in compact:
        aliases.update({compact.replace("-", ""), compact.replace("-", " ")})
    camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", compact)
    aliases.add(camel)
    initialism = compact.replace(".", "")
    if len(initialism) >= 2 and all(char.isupper() or char.isdigit() for char in initialism):
        aliases.add(" ".join(initialism))
    aliases.update(_LEGACY_ALIASES.get(canonical, []))
    return {alias.strip() for alias in aliases if alias.strip()}


def _ascii_pattern(alias: str) -> re.Pattern:
    tokens = alias.split()
    body = r"\s+".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


class HotwordProcessor:
    """Rewrite ASR output to user-approved canonical hotword spellings."""

    def __init__(self, hotwords: Iterable[str], *, fuzzy_pinyin_enabled: bool = True):
        self.hotwords = {str(word).strip() for word in hotwords if str(word).strip()}
        self.fuzzy_pinyin_enabled = bool(fuzzy_pinyin_enabled)
        self._ascii_rules: list[tuple[str, str, re.Pattern]] = []
        for canonical in self.hotwords:
            if not canonical.isascii():
                continue
            for alias in _ascii_aliases(canonical):
                self._ascii_rules.append((alias, canonical, _ascii_pattern(alias)))
        self._ascii_rules.sort(key=lambda item: len(item[0]), reverse=True)

        self._cjk_words: list[tuple[str, tuple[frozenset[str], ...]]] = []
        if self.fuzzy_pinyin_enabled:
            for word in self.hotwords:
                if len(word) < 2 or not _CJK.fullmatch(word):
                    continue
                readings = tuple(_character_readings(char) for char in word)
                if all(readings):
                    self._cjk_words.append((word, readings))
            self._cjk_words.sort(key=lambda item: len(item[0]), reverse=True)

    def rewrite(self, text: str) -> tuple[str, list[dict]]:
        output = str(text or "")
        corrections: list[dict] = []

        for alias, canonical, pattern in self._ascii_rules:
            def replace(match: re.Match) -> str:
                original = match.group(0)
                if original == canonical:
                    return original
                corrections.append({
                    "position": match.start(),
                    "end": match.end(),
                    "original": original,
                    "corrected": canonical,
                    "method": "canonical_alias",
                    "confidence": 0.95,
                })
                return canonical

            output = pattern.sub(replace, output)

        if not self._cjk_words:
            return output, corrections

        chars = list(output)
        char_readings = [_character_readings(char) for char in chars]
        rewritten: list[str] = []
        index = 0
        while index < len(chars):
            matched = False
            for word, target_readings in self._cjk_words:
                length = len(word)
                if index + length > len(chars):
                    continue
                source = "".join(chars[index:index + length])
                if source == word:
                    continue
                if all(
                    current and target and not current.isdisjoint(target)
                    for current, target in zip(
                        char_readings[index:index + length],
                        target_readings,
                    )
                ):
                    rewritten.append(word)
                    corrections.append({
                        "position": index,
                        "end": index + length,
                        "original": source,
                        "corrected": word,
                        "method": "fuzzy_pinyin",
                        "confidence": 0.9,
                    })
                    index += length
                    matched = True
                    break
            if not matched:
                rewritten.append(chars[index])
                index += 1
        return "".join(rewritten), corrections

    def matched_terms(self, text: str) -> list[str]:
        lowered = text.casefold()
        return sorted(
            (word for word in self.hotwords if word.casefold() in lowered),
            key=len,
            reverse=True,
        )
