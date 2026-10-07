"""LLM and retry helpers. The OpenAI adapter uses the current SDK (``openai>=1``):
``client.chat.completions.create`` and ``client.images.generate``.
"""
from __future__ import annotations

import importlib
import time
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


class ProviderError(RuntimeError):
    """A provider call failed after all retries."""


def with_retries(fn: Callable[[], T], attempts: int = 3, backoff_s: float = 1.0, sleep=time.sleep) -> T:
    """Call ``fn``. After a failure, wait ``backoff_s * 2**i`` and try again."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # provider SDKs raise many types
            last = exc
            if i < attempts - 1:
                sleep(backoff_s * 2**i)
    raise ProviderError(f"failed after {attempts} attempts: {last}") from last


class ChatLLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


def openai_client(api_key: str, timeout_s: float = 60.0):
    if not api_key:
        raise ProviderError("OPENAI_API_KEY is not set: put it in the environment or in a local .env file")
    openai = importlib.import_module("openai")  # optional extra "openai"
    return openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)


class OpenAIChat:
    def __init__(self, model: str, api_key: str = "", client=None, attempts: int = 3) -> None:
        self.client = client or openai_client(api_key)
        self.model = model
        self.name = f"openai:{model}"
        self.attempts = attempts

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        def call():
            reply = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                seed=0,
                max_tokens=max_tokens,
            )
            choice = reply.choices[0]
            if getattr(choice, "finish_reason", "stop") == "length":
                raise ProviderError("the reply was cut at max_tokens")
            return (choice.message.content or "").strip()

        return with_retries(call, attempts=self.attempts)


class ScriptedLLM:
    """Test and demo double: returns fixed replies in sequence and records the prompts."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append((system, user, max_tokens))
        return self.replies[min(len(self.calls), len(self.replies)) - 1]
