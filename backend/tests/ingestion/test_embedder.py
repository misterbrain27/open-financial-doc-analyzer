import httpx
import pytest

from app.ingestion import embedder


async def test_embed_texts_returns_embeddings():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 1024]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await embedder.embed_texts(["texte"], client=client)

    assert result == [[0.1] * 1024]


async def test_embed_texts_empty_list_returns_empty_without_request():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await embedder.embed_texts([], client=client)

    assert result == []


async def test_embed_texts_wrong_dimension_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 104]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError):
        await embedder.embed_texts(["texte"], client=client)
