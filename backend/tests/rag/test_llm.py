import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.rag.llm import GroqLLMClient, get_llm_client


class MockChoice:
    """Mock choice object for Groq streaming response."""

    def __init__(self, content: str | None):
        self.delta = Mock()
        self.delta.content = content


class MockChunk:
    """Mock chunk object for Groq streaming response."""

    def __init__(self, content: str | None):
        self.choices = [MockChoice(content)]


def make_groq_client(chunks: list[str]) -> Mock:
    """Creates a mock Groq client that streams the given chunks."""
    mock_client = Mock()

    # Simulate the streaming response
    def mock_stream_generator():
        for chunk in chunks:
            yield MockChunk(chunk)
        yield MockChunk(None)  # Final empty chunk

    mock_create = Mock(return_value=iter(mock_stream_generator()))
    mock_client.chat.completions.create = mock_create

    return mock_client


async def test_stream_chat_yields_text_deltas():
    client = GroqLLMClient(client=make_groq_client(["Bonjour", " le", " monde"]))

    deltas = [delta async for delta in client.stream_chat([{"role": "user", "content": "Salut"}])]

    assert deltas == ["Bonjour", " le", " monde"]


async def test_stream_chat_skips_empty_deltas():
    client = GroqLLMClient(client=make_groq_client(["", "Bonjour", ""]))

    deltas = [delta async for delta in client.stream_chat([{"role": "user", "content": "Salut"}])]

    assert deltas == ["Bonjour"]


def test_get_llm_client_returns_groq_client_by_default():
    assert isinstance(get_llm_client(), GroqLLMClient)


def test_get_llm_client_raises_for_unimplemented_provider(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "llm_provider", "ollama")

    with pytest.raises(NotImplementedError):
        get_llm_client()
