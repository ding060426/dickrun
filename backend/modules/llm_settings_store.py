"""User-level LLM settings persistence with simple API-key encryption."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from .record_store import connect, init_db, now_iso
from .llm_models import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, provider_defaults


_KEY_PATH = Path(
    os.environ.get(
        "DITING_LLM_KEY_FILE",
        str(Path(__file__).resolve().parents[1] / "data" / "llm_secret.key"),
    )
)


def _load_fernet() -> Fernet:
    configured = os.environ.get("DITING_LLM_ENCRYPTION_KEY", "").strip()
    if configured:
        key = base64.urlsafe_b64encode(hashlib.sha256(configured.encode("utf-8")).digest())
        return Fernet(key)
    if _KEY_PATH.is_file():
        return Fernet(_KEY_PATH.read_bytes().strip())
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    temporary = _KEY_PATH.with_suffix(_KEY_PATH.suffix + ".tmp")
    temporary.write_bytes(key)
    os.replace(temporary, _KEY_PATH)
    return Fernet(key)


_FERNET = _load_fernet()


DEFAULT_LLM_SETTINGS = {
    "provider": DEFAULT_PROVIDER,
    "base_url": provider_defaults(DEFAULT_PROVIDER)["base_url"],
    "model_name": DEFAULT_MODEL,
    "temperature": 0.2,
    "max_tokens": 8192,
    "timeout_sec": 180,
    "diagram_enabled": True,
    "diagram_type": "auto",
    "output_language": "zh-CN",
    "formal_style": True,
    "formula_mode": "latex",
}


def _encrypt(plain: str) -> str:
    if not plain:
        return ""
    return "fernet:" + _FERNET.encrypt(plain.encode("utf-8")).decode("ascii")


def _decrypt(encoded: str) -> str:
    if not encoded:
        return ""
    if not encoded.startswith("fernet:"):
        return ""
    try:
        return _FERNET.decrypt(encoded.removeprefix("fernet:").encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError):
        return ""


def _clamp_number(value, default, minimum, maximum, *, integer=False):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = min(float(maximum), max(float(minimum), parsed))
    return int(parsed) if integer else round(parsed, 2)


def default_settings() -> dict:
    return {**DEFAULT_LLM_SETTINGS, "has_api_key": False, "updated_at": ""}


def normalize_settings(data: dict | None) -> dict:
    source = data if isinstance(data, dict) else {}
    provider = str(source.get("provider") or DEFAULT_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        provider = "openai_compatible"
    provider_config = provider_defaults(provider)
    base_url = str(source.get("base_url") or provider_config["base_url"]).strip().rstrip("/")
    model_name = str(source.get("model_name") or provider_config["default_model"] or DEFAULT_MODEL).strip()
    diagram_type = str(source.get("diagram_type") or "auto").strip().lower()
    if diagram_type not in {"auto", "mindmap", "flowchart", "architecture"}:
        diagram_type = "auto"
    output_language = str(source.get("output_language") or "zh-CN")
    if output_language not in {"zh-CN", "en"}:
        output_language = "zh-CN"
    formula_mode = str(source.get("formula_mode") or "latex")
    if formula_mode not in {"latex", "plain"}:
        formula_mode = "latex"
    return {
        "provider": provider,
        "base_url": base_url[:2048],
        "model_name": model_name[:512],
        "temperature": _clamp_number(source.get("temperature"), 0.2, 0, 2),
        "max_tokens": _clamp_number(source.get("max_tokens"), 8192, 256, 384000, integer=True),
        "timeout_sec": _clamp_number(source.get("timeout_sec"), 180, 10, 600, integer=True),
        "diagram_enabled": bool(source.get("diagram_enabled", True)),
        "diagram_type": diagram_type,
        "output_language": output_language,
        "formal_style": bool(source.get("formal_style", True)),
        "formula_mode": formula_mode,
    }


def _settings_to_dict(row: Any) -> dict:
    return {
        "provider": row["provider"],
        "base_url": row["base_url"],
        "model_name": row["model_name"],
        "temperature": row["temperature"],
        "max_tokens": row["max_tokens"],
        "timeout_sec": row["timeout_sec"],
        "diagram_enabled": bool(row["diagram_enabled"]),
        "diagram_type": row["diagram_type"],
        "output_language": row["output_language"],
        "formal_style": bool(row["formal_style"]),
        "formula_mode": row["formula_mode"],
        "has_api_key": bool(row["api_key_encrypted"]),
        "updated_at": row["updated_at"],
    }


def get_settings(user_id: str) -> dict | None:
    """Return user LLM settings without the plaintext API key."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM user_llm_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    return _settings_to_dict(row)


def save_settings(user_id: str, data: dict) -> dict:
    """Save or update user LLM settings. Encrypts api_key if provided."""
    init_db()
    timestamp = now_iso()
    existing_settings = get_settings(user_id) or {}
    normalized = normalize_settings({**existing_settings, **(data or {})})
    with connect() as conn:
        existing = conn.execute(
            "SELECT api_key_encrypted FROM user_llm_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        api_key_encrypted = existing["api_key_encrypted"] if existing else ""
        if "api_key" in data:
            api_key_encrypted = _encrypt(data["api_key"]) if data["api_key"] else ""

        conn.execute(
            """
            INSERT INTO user_llm_settings (
                user_id, provider, base_url, api_key_encrypted, model_name,
                temperature, max_tokens, timeout_sec, diagram_enabled, diagram_type,
                output_language, formal_style, formula_mode, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                provider=excluded.provider,
                base_url=excluded.base_url,
                api_key_encrypted=excluded.api_key_encrypted,
                model_name=excluded.model_name,
                temperature=excluded.temperature,
                max_tokens=excluded.max_tokens,
                timeout_sec=excluded.timeout_sec,
                diagram_enabled=excluded.diagram_enabled,
                diagram_type=excluded.diagram_type,
                output_language=excluded.output_language,
                formal_style=excluded.formal_style,
                formula_mode=excluded.formula_mode,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                normalized["provider"],
                normalized["base_url"],
                api_key_encrypted,
                normalized["model_name"],
                normalized["temperature"],
                normalized["max_tokens"],
                normalized["timeout_sec"],
                int(normalized["diagram_enabled"]),
                normalized["diagram_type"],
                normalized["output_language"],
                int(normalized["formal_style"]),
                normalized["formula_mode"],
                timestamp,
            ),
        )
    return get_settings(user_id)


def get_decrypted_api_key(user_id: str) -> str:
    """Return the plaintext API key for use in LLM calls. Internal only."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT api_key_encrypted FROM user_llm_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row or not row["api_key_encrypted"]:
        return ""
    return _decrypt(row["api_key_encrypted"])


def get_effective_settings(user_id: str, override: dict | None = None) -> dict:
    """Merge: override > user saved settings > environment variables.
    Returns a dict suitable for constructing an LLMClient.
    """
    user = get_settings(user_id) or {}
    result = {
        **DEFAULT_LLM_SETTINGS,
        "api_key": "",
    }

    # Layer 1: env vars (lowest priority)
    from .llm_client import LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    if LLM_PROVIDER:
        result["provider"] = LLM_PROVIDER
    if LLM_BASE_URL:
        result["base_url"] = LLM_BASE_URL
    if LLM_API_KEY:
        result["api_key"] = LLM_API_KEY
    if LLM_MODEL:
        result["model_name"] = LLM_MODEL

    # Layer 2: user saved settings
    if user:
        if user.get("provider"):
            result["provider"] = user["provider"]
        if user.get("base_url"):
            result["base_url"] = user["base_url"]
        if user.get("model_name"):
            result["model_name"] = user["model_name"]
        result["temperature"] = user.get("temperature", 0.2)
        result["max_tokens"] = user.get("max_tokens", 8192)
        result["timeout_sec"] = user.get("timeout_sec", 180)
        result["diagram_enabled"] = user.get("diagram_enabled", True)
        result["diagram_type"] = user.get("diagram_type", "auto")
        result["output_language"] = user.get("output_language", "zh-CN")
        result["formal_style"] = user.get("formal_style", True)
        result["formula_mode"] = user.get("formula_mode", "latex")
        # Decrypt api_key
        decrypted = get_decrypted_api_key(user_id)
        if decrypted:
            result["api_key"] = decrypted

    # Layer 3: per-request override (highest priority)
    if override:
        for k in ("provider", "base_url", "api_key", "model_name", "temperature",
                   "max_tokens", "timeout_sec", "diagram_enabled", "diagram_type",
                   "output_language", "formal_style", "formula_mode"):
            if k not in override or override[k] is None:
                continue
            if k in {"provider", "base_url", "api_key", "model_name"} and not str(override[k]).strip():
                continue
            result[k] = override[k]

    normalized = normalize_settings(result)
    return {**normalized, "api_key": str(result.get("api_key") or "")}


def public_effective_settings(user_id: str) -> dict:
    effective = get_effective_settings(user_id)
    return {
        **{key: value for key, value in effective.items() if key != "api_key"},
        "has_api_key": bool(effective.get("api_key")),
        "updated_at": (get_settings(user_id) or {}).get("updated_at", ""),
    }
