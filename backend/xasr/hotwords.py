"""Prepare sherpa-onnx contextual-biasing assets from user hotwords."""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


_SCORE_SUFFIX = re.compile(r"\s:\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class HotwordAssets:
    decoding_method: str
    hotwords_file: Path | None = None
    bpe_vocab: Path | None = None
    modeling_unit: str = "cjkchar"
    max_active_paths: int = 4

    @property
    def enabled(self) -> bool:
        return self.hotwords_file is not None and self.bpe_vocab is not None


def prepare_hotword_assets(
    model_dir: str | Path,
    hotwords: list[str],
    *,
    score: float = 5.0,
    scores: dict[str, float] | None = None,
    runtime_dir: str | Path | None = None,
) -> HotwordAssets:
    normalized = [word.strip() for word in hotwords if word and word.strip()]
    tokens_path = Path(model_dir) / "tokens.txt"
    if not normalized or not tokens_path.is_file():
        return HotwordAssets(decoding_method="greedy_search")

    output_dir = Path(runtime_dir or Path(tempfile.gettempdir()) / "huiwu-xasr")
    output_dir.mkdir(parents=True, exist_ok=True)
    token_bytes = tokens_path.read_bytes()
    token_hash = hashlib.sha256(token_bytes).hexdigest()[:12]
    bpe_vocab = output_dir / f"bpe-{token_hash}.vocab"
    if not bpe_vocab.is_file():
        _write_bpe_vocab(token_bytes.decode("utf-8"), bpe_vocab)

    prepared_lines = [
        variant
        for word in normalized
        for variant in _expand_hotword(word, (scores or {}).get(word, score), explicit=word in (scores or {}))
    ]
    hotword_payload = ("\n".join(prepared_lines) + "\n").encode("utf-8")
    hotword_hash = hashlib.sha256(hotword_payload).hexdigest()[:12]
    hotwords_file = output_dir / f"hotwords-{hotword_hash}.txt"
    if not hotwords_file.is_file():
        hotwords_file.write_bytes(hotword_payload)

    return HotwordAssets(
        decoding_method="modified_beam_search",
        hotwords_file=hotwords_file,
        bpe_vocab=bpe_vocab,
        modeling_unit="bpe",
    )


def _is_cjk(character: str) -> bool:
    value = ord(character)
    return (
        0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
    )


def _space_cjk(value: str) -> str:
    expanded = "".join(f" {char} " if _is_cjk(char) else char for char in value)
    return " ".join(expanded.split())


def _format_score(score: float) -> str:
    return str(int(score)) if score == int(score) else f"{score:.3f}".rstrip("0").rstrip(".")


def _with_boost(value: str, score: float, *, explicit: bool = False) -> str:
    if _SCORE_SUFFIX.search(value):
        match = _SCORE_SUFFIX.search(value)
        return f"{_space_cjk(value[:match.start()])} {match.group(0).strip()}"
    applied = score if explicit or any(_is_cjk(char) for char in value) else min(score, 2.5)
    return f"{_space_cjk(value)} :{_format_score(applied)}"


def _expand_hotword(value: str, score: float, *, explicit: bool = False) -> list[str]:
    if _SCORE_SUFFIX.search(value) or any(_is_cjk(char) for char in value):
        return [_with_boost(value, score, explicit=explicit)]
    capitalized = value[:1].upper() + value[1:]
    variants = [value] if capitalized == value else [value, capitalized]
    return [_with_boost(variant, score, explicit=explicit) for variant in variants]


def _write_bpe_vocab(tokens_text: str, output_path: Path) -> None:
    rows: list[str] = []
    pieces: set[str] = set()
    bare_cjk: set[str] = set()
    marker = "▁"
    for line in tokens_text.splitlines():
        try:
            piece, raw_index = line.rsplit(" ", 1)
            index = int(raw_index)
        except (ValueError, TypeError):
            continue
        rows.append(f"{piece}\t{-index}")
        pieces.add(piece)
        if len(piece) == 2 and piece[0] == marker and _is_cjk(piece[1]):
            bare_cjk.add(piece[1])
    if marker not in pieces:
        rows.append(f"{marker}\t-1")
    rows.extend(f"{character}\t-999999" for character in sorted(bare_cjk - pieces))
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
