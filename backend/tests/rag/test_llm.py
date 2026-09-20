import json

import httpx
import pytest
from mistralai.client import Mistral

from app.rag.llm import MistralLLMClient, get_llm_client


def sse_response(chunks: list[str]) -> httpx.Response:
    """Builds a minimal SSE response as expected by `mistralai` (streaming):
    one `data: <json>` line per event, terminated by the `data: [DONE]` sentinel."""
    events = [
        {
            "id": "cmpl-test",
            "model": "mistral-small-latest",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        for content in chunks
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def make_mistral_client(chunks: list[str]) -> Mistral:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chunks)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Mistral(api_key="test-key", async_client=async_client)


async def test_stream_chat_yields_text_deltas():
    client = MistralLLMClient(client=make_mistral_client(["Bonjour", " le", " monde"]))

    deltas = [delta async for delta in client.stream_chat([{"role": "user", "content": "Salut"}])]

    assert deltas == ["Bonjour", " le", " monde"]


async def test_stream_chat_skips_empty_deltas():
    client = MistralLLMClient(client=make_mistral_client(["", "Bonjour", ""]))

    deltas = [delta async for delta in client.stream_chat([{"role": "user", "content": "Salut"}])]

    assert deltas == ["Bonjour"]


def test_get_llm_client_returns_mistral_client_by_default():
    assert isinstance(get_llm_client(), MistralLLMClient)


def test_get_llm_client_raises_for_unimplemented_provider(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "llm_provider", "ollama")

    with pytest.raises(NotImplementedError):
        get_llm_client()
