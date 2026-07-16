"""Curated text-model presets and capability routing for meeting summaries."""

from __future__ import annotations


DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"


PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "requires_api_key": True,
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "requires_api_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "requires_api_key": True,
    },
    "ollama": {
        "label": "Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "requires_api_key": False,
    },
    "openai_compatible": {
        "label": "OpenAI Compatible / Custom",
        "base_url": "",
        "requires_api_key": True,
    },
}


_MODELS = (
    {
        "provider": "deepseek",
        "id": "deepseek-v4-flash",
        "label": "DSv4 Flash（默认 / 高性价比）",
    },
    {
        "provider": "deepseek",
        "id": "deepseek-v4-pro",
        "label": "DSv4 Pro（高质量）",
    },
    {
        "provider": "qwen",
        "id": "qwen3.7-max",
        "label": "Qwen3.7 Max",
    },
    {
        "provider": "qwen",
        "id": "qwen3.7-plus",
        "label": "Qwen3.7 Plus",
    },
    {
        "provider": "qwen",
        "id": "qwen3.6-flash",
        "label": "Qwen3.6 Flash",
    },
    {
        "provider": "openai",
        "id": "gpt-5.2",
        "label": "GPT-5.2",
    },
    {
        "provider": "openai",
        "id": "gpt-5.1",
        "label": "GPT-5.1",
    },
    {
        "provider": "openai",
        "id": "gpt-5-mini",
        "label": "GPT-5 mini",
    },
)


def provider_defaults(provider: str | None) -> dict:
    selected = str(provider or DEFAULT_PROVIDER).strip().lower()
    details = PROVIDERS.get(selected, PROVIDERS["openai_compatible"])
    default_model = next(
        (item["id"] for item in _MODELS if item["provider"] == selected),
        "",
    )
    return {
        "provider": selected if selected in PROVIDERS else "openai_compatible",
        **details,
        "default_model": default_model,
    }


def model_catalog(provider: str | None = None) -> list[dict]:
    """Return text-generation models known to work with this summary pipeline."""

    selected = str(provider or "").strip().lower()
    return [
        {
            **item,
            "image_generation": False,
            "diagram_mode": "text",
        }
        for item in _MODELS
        if not selected or item["provider"] == selected
    ]


def model_capabilities(model_id: str) -> dict:
    """Meeting diagrams are always textual Mermaid/Markdown, never image calls."""

    known = next((item for item in model_catalog() if item["id"] == model_id), None)
    if known:
        return {
            "image_generation": known["image_generation"],
            "diagram_mode": known["diagram_mode"],
        }
    return {"image_generation": False, "diagram_mode": "text"}


def public_catalog() -> dict:
    return {
        "default_provider": DEFAULT_PROVIDER,
        "default_model": DEFAULT_MODEL,
        "providers": [
            {"id": provider_id, **details}
            for provider_id, details in PROVIDERS.items()
        ],
        "models": model_catalog(),
    }
