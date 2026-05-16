"""
LangChain-style LLM Abstraction
Supports: Anthropic, OpenAI, Custom API
"""
import json
import asyncio
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator
from .messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, messages_to_dicts


@dataclass
class LLMResult:
    """Result returned from an LLM call."""
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = "end_turn"

    def __str__(self) -> str:
        return self.content


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._callbacks  = []

    def bind(self, **kwargs) -> "BaseLLM":
        """Return a copy of this LLM with bound parameters."""
        import copy
        new = copy.copy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new

    def with_config(self, **kwargs) -> "BaseLLM":
        return self.bind(**kwargs)

    @abstractmethod
    async def _acall(
        self,
        messages: List[BaseMessage],
        system: Optional[str] = None,
    ) -> LLMResult:
        pass

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        system: Optional[str] = None,
    ) -> LLMResult:
        result = await self._acall(messages, system)
        return result

    def invoke(
        self,
        messages: List[BaseMessage],
        system: Optional[str] = None,
    ) -> LLMResult:
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, system)
        )

    def _make_request(self, url: str, headers: Dict, body: Dict) -> Dict:
        """Sync HTTP POST using urllib (no external deps)."""
        data = json.dumps(body).encode("utf-8")
        req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8")
            raise RuntimeError(f"LLM API Error [{e.code}]: {body_text[:300]}")
        except Exception as e:
            raise RuntimeError(f"LLM Request Failed: {e}")

    async def _amake_request(self, url: str, headers: Dict, body: Dict) -> Dict:
        """Async HTTP POST (runs sync in thread executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._make_request, url, headers, body)

    def __or__(self, other):
        from .chains import LLMChain
        if callable(getattr(other, 'invoke', None)):
            from .prompts import _PipelineChain
            return _PipelineChain(self, other)
        return NotImplemented


# ─────────────────────────────────────────────
# Anthropic Provider
# ─────────────────────────────────────────────
class ChatAnthropic(BaseLLM):
    """Anthropic Claude API provider."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    API_URL       = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.api_key = api_key

    async def _acall(
        self,
        messages: List[BaseMessage],
        system: Optional[str] = None,
    ) -> LLMResult:
        # Separate system messages
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        chat_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        system_content = system or (sys_msgs[0].content if sys_msgs else "You are a helpful assistant.")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_content,
            "messages": messages_to_dicts(chat_msgs),
        }
        data = await self._amake_request(self.API_URL, headers, body)
        content = data["content"][0]["text"]
        return LLMResult(
            content=content,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            raw=data,
            stop_reason=data.get("stop_reason", "end_turn"),
        )


# ─────────────────────────────────────────────
# OpenAI Provider
# ─────────────────────────────────────────────
class ChatOpenAI(BaseLLM):
    """OpenAI ChatGPT API provider."""

    DEFAULT_MODEL = "gpt-4o"
    API_URL       = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.api_key = api_key

    async def _acall(
        self,
        messages: List[BaseMessage],
        system: Optional[str] = None,
    ) -> LLMResult:
        all_msgs = list(messages)
        if system:
            all_msgs = [SystemMessage(content=system)] + [
                m for m in all_msgs if not isinstance(m, SystemMessage)
            ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages_to_dicts(all_msgs),
        }
        data = await self._amake_request(self.API_URL, headers, body)
        content = data["choices"][0]["message"]["content"]
        return LLMResult(
            content=content,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            raw=data,
            stop_reason=data["choices"][0].get("finish_reason", "stop"),
        )


# ─────────────────────────────────────────────
# Custom / Generic OpenAI-compatible Provider
# ─────────────────────────────────────────────
class ChatCustom(BaseLLM):
    """Generic OpenAI-compatible API provider."""

    def __init__(self, api_key: str, base_url: str, model: str = "custom-model", **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")

    async def _acall(
        self,
        messages: List[BaseMessage],
        system: Optional[str] = None,
    ) -> LLMResult:
        all_msgs = list(messages)
        if system:
            all_msgs = [SystemMessage(content=system)] + [
                m for m in all_msgs if not isinstance(m, SystemMessage)
            ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages_to_dicts(all_msgs),
        }
        url = f"{self.base_url}/chat/completions"
        data = await self._amake_request(url, headers, body)
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content")
            or data.get("content", [{}])[0].get("text", "")
        )
        return LLMResult(content=content, model=self.model, raw=data)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def create_llm(
    provider: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
    **kwargs
) -> BaseLLM:
    """Factory function to create the correct LLM provider."""
    provider = provider.lower()
    if provider == "anthropic":
        return ChatAnthropic(api_key=api_key, model=model, **kwargs)
    elif provider == "openai":
        return ChatOpenAI(api_key=api_key, model=model, **kwargs)
    elif provider == "custom":
        if not base_url:
            raise ValueError("base_url is required for custom provider")
        return ChatCustom(api_key=api_key, base_url=base_url, model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose: anthropic, openai, custom")
