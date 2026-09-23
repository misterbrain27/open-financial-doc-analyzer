"""LLM interface + Groq implementation (Phase 5).

`LLMClient` decouples `rag/pipeline.py` from the generation provider (strategy / dependency
inversion): switching to Ollama (offline, for confidential use cases — see `doc/plan.md`)
won't touch the pipeline, only a new implementation will be added here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from groq import Groq

from app.config import settings


class LLMClient(Protocol):
    def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Generates a response and yields text deltas as generation proceeds."""
        ...


class GroqLLMClient:
    """Groq implementation (`groq`), streaming via Server-Sent Events."""

    def __init__(self, *, client: Groq | None = None) -> None:
        self._client = client or Groq(api_key=settings.groq_api_key)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        # Groq client doesn't have native async support, so we use sync client in executor
        import asyncio

        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                stream=True,
            ),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            # `delta` may be `None` (e.g. first event, just the role): filter it out.
            if isinstance(delta, str) and delta:
                yield delta


def get_llm_client() -> LLMClient:
    """Selects the implementation based on `settings.llm_provider` ('groq' | 'ollama')."""
    if settings.llm_provider == "groq":
        return GroqLLMClient()
    raise NotImplementedError(
        f"Fournisseur LLM {settings.llm_provider!r} non implémenté (seul 'groq' l'est actuellement)"
    )
