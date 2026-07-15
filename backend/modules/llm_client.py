"""
谛听 DiTing - LLM API 抽象层
============================================================================
统一的 LLM API 调用接口，支持多个提供商，自动降级。

提供商:
  - GPT55:     apifusion.aispeech.com.cn (默认，OpenAI 兼容)
  - DeepSeek:  api.deepseek.com/v1/chat/completions (推荐，¥0.1/万token)
  - Qwen:      dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
  - OpenAI:    api.openai.com/v1/chat/completions (兼容接口)
  - Mock:      离线兜底（返回空/规则结果）

配置 (环境变量，可选覆盖默认值):
  DITING_LLM_PROVIDER  — gpt55 | deepseek | qwen | openai | mock (默认 gpt55)
  DITING_LLM_API_KEY   — API Key
  DITING_LLM_MODEL     — 模型名 (默认: gpt5.5)
  DITING_LLM_BASE_URL  — 自定义 API 地址

用法:
    from modules.llm_client import get_llm_client

    client = get_llm_client()
    result = client.chat("介绍一下北京", system="你是旅游助手")
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger("llm_client")


# ======================================================================
# 配置
# ======================================================================

def _env(name: str, default: str = None) -> Optional[str]:
    return os.environ.get(f"DITING_LLM_{name}", os.environ.get(name, default))


LLM_PROVIDER = _env("PROVIDER", "gpt55").lower()
LLM_API_KEY = _env("API_KEY", "")
LLM_MODEL = _env("MODEL", "gpt-5.5")
LLM_BASE_URL = _env("BASE_URL", "https://apifusion.aispeech.com.cn")


# ======================================================================
# 抽象基类
# ======================================================================

class LLMClient(ABC):
    """LLM API 抽象基类"""

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """单轮对话，返回文本回复。"""

    @abstractmethod
    def chat_json(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.1,
    ) -> Optional[dict]:
        """单轮对话，返回 JSON 解析结果。失败返回 None。"""

    def classify(self, text: str, labels: List[str], description: str = "") -> str:
        """分类任务：将 text 归入 labels 之一。"""
        label_list = "\n".join(f"- {l}" for l in labels)
        prompt = f"""请将以下文本归入最合适的类别。

{description}

文本: {text}

类别:
{label_list}

请只返回最匹配的一个类别名称，不要解释。"""
        result = self.chat(prompt, system="你是文本分类专家。", temperature=0.0)
        result = result.strip().strip("'\"").strip()
        # 精确匹配
        for label in labels:
            if label in result:
                return label
        return labels[0] if labels else ""

    def extract_keywords(self, text: str, max_count: int = 20) -> List[str]:
        """从文本中提取关键词。"""
        prompt = f"""请从以下文本中提取 {max_count} 个以内的关键术语和专有名词。

文本:
{text}

要求:
- 优先提取专业术语、产品名、指标名、技术缩写
- 每个关键词 2-10 个字
- 不要包含通用词汇
- 返回 JSON 数组格式

只返回 JSON 数组，不要其他内容。"""
        result = self.chat(prompt, system="你是关键词提取专家。", temperature=0.0)
        try:
            # 尝试从回复中提取 JSON 数组
            result = result.strip()
            if result.startswith("```"):
                # 移除 markdown 代码块标记
                lines = result.split("\n")
                result = "\n".join(l for l in lines if not l.startswith("```"))
            keywords = json.loads(result)
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords[:max_count] if str(k).strip()]
        except json.JSONDecodeError:
            # 非 JSON 格式，尝试按行分割
            lines = [l.strip().strip("-,.，。1234567890.、") for l in result.split("\n")]
            return [l for l in lines if l and len(l) >= 2][:max_count]
        return []

    def summarize(self, transcript: List[dict], meeting_title: str = "") -> dict:
        """生成会议摘要。子类可覆盖以优化。"""
        from .domain_taxonomy import DOMAIN_TAXONOMY  # 延迟导入避免循环

        # 构建转写文本
        transcript_text = ""
        for seg in transcript:
            speaker = seg.get("speaker", "未知")
            text = seg.get("text", "")
            transcript_text += f"[{speaker}]: {text}\n"

        prompt = f"""你是一位专业的会议记录员。请根据以下会议转写内容，生成结构化的会议纪要。

会议标题: {meeting_title or '未命名会议'}

会议转写:
{transcript_text}

请返回 JSON 格式（确保是合法 JSON）:
{{
  "summary": "会议总体概述 (2-3句)",
  "topics": ["讨论主题1", "讨论主题2"],
  "decisions": [{{"decision": "决策内容", "by": "决策人"}}],
  "action_items": [{{"task": "待办事项", "assignee": "负责人", "deadline": "时间或待定"}}],
  "key_metrics": [{{"metric": "指标名", "value": "数值", "context": "上下文"}}],
  "data_conflicts": [{{"description": "冲突描述", "resolution": "解决情况"}}],
  "next_meeting": "建议下次会议时间/议题或空"
}}

只返回 JSON，不要有其他文字。"""

        result = self.chat(prompt, system="你是专业的会议记录和摘要生成助手。", temperature=0.2)
        try:
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(l for l in lines if not l.startswith("```"))
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM summary JSON. Raw: {result[:200]}")
            return {
                "summary": result[:200] if result else "",
                "topics": [], "decisions": [], "action_items": [],
                "key_metrics": [], "data_conflicts": [], "next_meeting": "",
            }

    @property
    def is_available(self) -> bool:
        """LLM API 是否可用。"""
        return True


# ======================================================================
# DeepSeek 客户端
# ======================================================================

class DeepSeekClient(LLMClient):
    """DeepSeek API 客户端 (兼容 OpenAI SDK)"""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or LLM_API_KEY
        self.model = model or LLM_MODEL
        self.base_url = base_url or LLM_BASE_URL or "https://api.deepseek.com"
        self._available = bool(self.api_key)
        if not self._available:
            logger.warning("DeepSeek API key not set. Set DITING_LLM_API_KEY env var.")

    def _call_api(self, messages: List[dict], temperature: float, max_tokens: int) -> Optional[str]:
        """通过 httpx 调用 DeepSeek API (OpenAI 兼容格式)。"""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.warning(f"DeepSeek API error {resp.status_code}: {resp.text[:300]}")
                    return None
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"DeepSeek API call failed: {e}")
            return None

    def chat(self, prompt: str, system: str = None, temperature: float = 0.3,
             max_tokens: int = 2048) -> str:
        if not self._available:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._call_api(messages, temperature, max_tokens)
        return result or ""

    def chat_json(self, prompt: str, system: str = None,
                  temperature: float = 0.1) -> Optional[dict]:
        if not self._available:
            return None
        full_prompt = prompt + "\n\n只返回合法的 JSON，不要有任何其他文字。"
        result = self.chat(full_prompt, system=system, temperature=temperature)
        if not result:
            return None
        try:
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(l for l in lines if not l.startswith("```"))
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from DeepSeek response")
            return None

    @property
    def is_available(self) -> bool:
        return self._available


# ======================================================================
# Qwen (通义千问) 客户端
# ======================================================================

class QwenClient(LLMClient):
    """阿里云通义千问 API 客户端 (OpenAI 兼容模式)"""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or LLM_API_KEY
        self.model = model or LLM_MODEL or "qwen-turbo"
        self.base_url = (
            base_url or LLM_BASE_URL
            or "https://dashscope.aliyuncs.com/compatible-mode"
        )
        self._available = bool(self.api_key)
        if not self._available:
            logger.warning("Qwen API key not set. Set DITING_LLM_API_KEY env var.")

    def _call_api(self, messages: List[dict], temperature: float, max_tokens: int) -> Optional[str]:
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.warning(f"Qwen API error {resp.status_code}: {resp.text[:300]}")
                    return None
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Qwen API call failed: {e}")
            return None

    def chat(self, prompt: str, system: str = None, temperature: float = 0.3,
             max_tokens: int = 2048) -> str:
        if not self._available:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._call_api(messages, temperature, max_tokens)
        return result or ""

    def chat_json(self, prompt: str, system: str = None,
                  temperature: float = 0.1) -> Optional[dict]:
        if not self._available:
            return None
        full_prompt = prompt + "\n\n只返回合法的 JSON，不要有任何其他文字。"
        result = self.chat(full_prompt, system=system, temperature=temperature)
        if not result:
            return None
        try:
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(l for l in lines if not l.startswith("```"))
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Qwen response")
            return None

    @property
    def is_available(self) -> bool:
        return self._available


# ======================================================================
# OpenAI 客户端 (通用兼容)
# ======================================================================

class OpenAIClient(LLMClient):
    """OpenAI / 兼容接口 客户端"""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or LLM_API_KEY
        self.model = model or LLM_MODEL or "gpt-4o-mini"
        self.base_url = base_url or LLM_BASE_URL or "https://api.openai.com"
        self._available = bool(self.api_key)
        if not self._available:
            logger.warning("OpenAI API key not set. Set DITING_LLM_API_KEY env var.")

    def _call_api(self, messages: List[dict], temperature: float, max_tokens: int) -> Optional[str]:
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return None

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.warning(f"OpenAI API error {resp.status_code}: {resp.text[:300]}")
                    return None
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            return None

    def chat(self, prompt: str, system: str = None, temperature: float = 0.3,
             max_tokens: int = 2048) -> str:
        if not self._available:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._call_api(messages, temperature, max_tokens)
        return result or ""

    def chat_json(self, prompt: str, system: str = None,
                  temperature: float = 0.1) -> Optional[dict]:
        if not self._available:
            return None
        full_prompt = prompt + "\n\n只返回合法的 JSON，不要有任何其他文字。"
        result = self.chat(full_prompt, system=system, temperature=temperature)
        if not result:
            return None
        try:
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(l for l in lines if not l.startswith("```"))
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from OpenAI response")
            return None

    @property
    def is_available(self) -> bool:
        return self._available


# ======================================================================
# GPT5.5 客户端 (aispeech fusion)
# ======================================================================

class GPT55Client(OpenAIClient):
    """GPT-5.5 via AISpeech Fusion — OpenAI 兼容接口。"""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        super().__init__(
            api_key=api_key or LLM_API_KEY,
            model=model or LLM_MODEL,
            base_url=base_url or LLM_BASE_URL,
        )


# ======================================================================
# Mock 客户端 (离线兜底)
# ======================================================================

class MockClient(LLMClient):
    """离线 Mock 客户端 — 所有方法返回空结果，系统以规则模式运行。"""

    def chat(self, prompt: str, system: str = None, temperature: float = 0.3,
             max_tokens: int = 2048) -> str:
        return ""

    def chat_json(self, prompt: str, system: str = None,
                  temperature: float = 0.1) -> Optional[dict]:
        return None

    def classify(self, text: str, labels: List[str], description: str = "") -> str:
        return labels[0] if labels else ""

    def extract_keywords(self, text: str, max_count: int = 20) -> List[str]:
        return []

    def summarize(self, transcript: List[dict], meeting_title: str = "") -> dict:
        return {
            "summary": "",
            "topics": [], "decisions": [], "action_items": [],
            "key_metrics": [], "data_conflicts": [], "next_meeting": "",
        }

    @property
    def is_available(self) -> bool:
        return False


# ======================================================================
# 工厂函数
# ======================================================================

_CLIENT_REGISTRY = {
    "gpt55": GPT55Client,
    "deepseek": DeepSeekClient,
    "qwen": QwenClient,
    "openai": OpenAIClient,
    "mock": MockClient,
}

_llm_client: Optional[LLMClient] = None


def get_llm_client(provider: str = None) -> LLMClient:
    """获取 LLM 客户端实例 (单例)。

    Args:
        provider: deepseek | qwen | openai | mock (默认从环境变量读取)
    """
    global _llm_client

    provider = (provider or LLM_PROVIDER).lower()

    if _llm_client is not None and _llm_client.__class__.__name__.lower().startswith(provider):
        return _llm_client

    client_cls = _CLIENT_REGISTRY.get(provider, MockClient)
    if client_cls is MockClient or provider not in _CLIENT_REGISTRY:
        logger.info(f"LLM provider set to 'mock' — cognitive features will use rule-based fallback")
        logger.info(f"To enable LLM, set: DITING_LLM_PROVIDER=deepseek DITING_LLM_API_KEY=your-key")

    _llm_client = client_cls()
    return _llm_client


def reset_llm_client():
    """重置 LLM 客户端 (用于测试或切换提供商)。"""
    global _llm_client
    _llm_client = None
