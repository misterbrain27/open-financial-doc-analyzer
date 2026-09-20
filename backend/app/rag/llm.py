"""LLM interface + Mistral implementation (Phase 5).

`LLMClient` decouples `rag/pipeline.py` from the generation provider (strategy / dependency
inversion): switching to Ollama (offline, for confidential use cases — see `doc/plan.md`)
won't touch the pipeline, only a new implementation will be added here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from mistralai.client import Mistral

from app.config import settings


class LLMClient(Protocol):
    def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Generates a response and yields text deltas as generation proceeds."""
        ...


class MistralLLMClient:
    """Mistral AI implementation (`mistralai`), streaming via Server-Sent Events."""

    def __init__(self, *, client: Mistral | None = None) -> None:
        self._client = client or Mistral(api_key=settings.mistral_api_key)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self._client.chat.stream_async(
            model=settings.mistral_model, messages=messages
        )
        async for event in stream:
            delta = event.data.choices[0].delta.content
            # `delta` may be `None`/absent (e.g. first event, just the role): filter it out.
            if isinstance(delta, str) and delta:
                yield delta


def get_llm_client() -> LLMClient:
    """Selects the implementation based on `settings.llm_provider` ('mistral' | 'ollama')."""
    if settings.llm_provider == "mistral":
        return MistralLLMClient()
    raise NotImplementedError(
        f"Fournisseur LLM {settings.llm_provider!r} non implémenté "
        "(seul 'mistral' l'est actuellement)"
    )
