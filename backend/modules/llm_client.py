"""Async OpenAI-compatible LLM client with retry, schema validation, and env-var config."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from .llm_models import provider_defaults

logger = logging.getLogger("llm_client")

# ── Configuration (env vars) ────────────────────────────────────

LLM_PROVIDER = os.environ.get("DITING_LLM_PROVIDER", "").strip()
LLM_BASE_URL = os.environ.get("DITING_LLM_BASE_URL", "").strip().rstrip("/")
LLM_API_KEY = os.environ.get("DITING_LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("DITING_LLM_MODEL", "").strip()
LLM_TIMEOUT_SEC = int(os.environ.get("DITING_LLM_TIMEOUT_SEC", "120"))
LLM_MAX_RETRIES = int(os.environ.get("DITING_LLM_MAX_RETRIES", "3"))
LLM_MAX_CONCURRENCY = int(os.environ.get("DITING_LLM_MAX_CONCURRENCY", "2"))


def _redact_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "***" + key[-4:]


class LLMNotConfiguredError(Exception):
    """Raised when LLM env vars are missing."""


class LLMResponseError(Exception):
    """Raised when LLM returns an unexpected response or unparseable JSON."""


class LLMClient:
    """Async OpenAI-compatible LLM client with retry and JSON Schema validation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: int | None = None,
        max_retries: int | None = None,
        max_concurrency: int | None = None,
        temperature: float | None = None,
        default_max_tokens: int | None = None,
        provider: str | None = None,
    ):
        self.provider = str(provider or LLM_PROVIDER or "openai_compatible").strip().lower()
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or LLM_API_KEY
        self.model = model or LLM_MODEL
        self.timeout_sec = timeout_sec or LLM_TIMEOUT_SEC
        self.max_retries = max_retries or LLM_MAX_RETRIES
        self.max_concurrency = max_concurrency or LLM_MAX_CONCURRENCY
        self.temperature = 0.2 if temperature is None else float(temperature)
        self.default_max_tokens = int(default_max_tokens or 8192)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._client: Any = None

    @classmethod
    def from_settings(cls, settings: dict) -> "LLMClient":
        return cls(
            base_url=settings.get("base_url"),
            api_key=settings.get("api_key"),
            model=settings.get("model_name"),
            timeout_sec=settings.get("timeout_sec"),
            temperature=settings.get("temperature"),
            default_max_tokens=settings.get("max_tokens"),
            provider=settings.get("provider"),
        )

    @property
    def is_configured(self) -> bool:
        requires_key = provider_defaults(self.provider)["requires_api_key"]
        return bool(self.base_url and self.model and (self.api_key or not requires_key))

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout_sec),
                    headers=headers,
                )
            except ImportError:
                raise LLMNotConfiguredError("httpx is not installed; run: pip install httpx")
        return self._client

    async def check_available(self) -> bool:
        """Return True if the LLM endpoint is reachable and configured."""
        if not self.is_configured:
            return False
        try:
            client = self._get_client()
            resp = await client.get("/models", timeout=10)
            return resp.status_code < 500
        except Exception:
            return False

    async def generate_json(
        self,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
        json_schema: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Send a chat-completion request and return validated JSON.

        Raises LLMNotConfiguredError if env vars are missing.
        Raises LLMResponseError on repeated failure.
        """
        if not self.is_configured:
            raise LLMNotConfiguredError(
                "LLM not configured. Set DITING_LLM_BASE_URL, DITING_LLM_API_KEY, DITING_LLM_MODEL."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        request_max_tokens = min(
            int(max_tokens or self.default_max_tokens),
            self.default_max_tokens,
        )
        request_temperature = self.temperature if temperature is None else float(temperature)

        logger.info(
            "LLM request: model=%s endpoint=%s key=%s tokens=%d",
            self.model, self.base_url, _redact_key(self.api_key), request_max_tokens,
        )

        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    client = self._get_client()
                    resp = await client.post(
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": request_max_tokens,
                            "temperature": request_temperature,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning("LLM 429 rate-limited, retry %d/%d after %ds", attempt, self.max_retries, wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    body = resp.json()
                    content = body["choices"][0]["message"]["content"]
                    result = self._parse_json(content)
                    if json_schema:
                        result = self._validate_schema(result, json_schema)
                    logger.info("LLM response OK: %d keys", len(result))
                    return result
                except (LLMResponseError, Exception) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        wait = 2 ** attempt
                        logger.warning("LLM error (attempt %d/%d): %s — retrying in %ds", attempt, self.max_retries, exc, wait)
                        await asyncio.sleep(wait)
                    else:
                        logger.error("LLM failed after %d attempts: %s", self.max_retries, exc)

        raise LLMResponseError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    def _parse_json(self, content: str) -> dict:
        # Some models wrap JSON in ```json fences
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM returned unparseable JSON: {exc}\nContent: {content[:500]}")

    def _validate_schema(self, data: dict, schema: dict) -> dict:
        try:
            import jsonschema
            jsonschema.validate(instance=data, schema=schema)
        except ImportError:
            logger.warning("jsonschema not installed — skipping schema validation")
        except jsonschema.ValidationError as exc:
            raise LLMResponseError(f"LLM JSON failed schema validation: {exc.message}")
        return data

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# Module-level singleton (created lazily)
_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
