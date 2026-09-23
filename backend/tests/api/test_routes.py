"""Tests for the API routes (Phase 6): `/ingest` and `/query`.

`db_session` (transactional fixture, cf. `tests/conftest.py`) is injected in place of
`get_session` via `app.dependency_overrides`. We use `httpx.AsyncClient` + `ASGITransport`
rather than `fastapi.testclient.TestClient`: these tests are coroutines that run in the same
asyncio loop as `db_session`/the SQLAlchemy engine — a synchronous `TestClient` runs the app
in a separate loop (anyio portal) and would break asyncpg (cf. the comment on
`asyncio_default_fixture_loop_scope` in `pyproject.toml`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.ingestion import pipeline as ingestion_pipeline_module
from app.ingestion.embedder import EMBEDDING_DIM
from app.main import app
from app.models import Chunk, Document
from app.rag import pipeline as rag_pipeline_module
from app.rag.llm import LLMClient
from app.retrieval import search as search_module


@pytest.fixture
async def api_client(db_session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override_get_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def make_pdf(path: Path, *, text: str = "Chiffre d'affaires 42M€") -> Path:
    """Generates a minimal PDF on the fly (same trick as `tests/ingestion/test_loader.py`)."""
    c = canvas.Canvas(str(path))
    c.drawString(72, 800, text)
    c.showPage()
    c.save()
    return path


def unit_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


# --- /ingest -----------------------------------------------------------------------------


async def test_ingest_rejects_non_pdf(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/ingest", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


async def test_ingest_saves_file_and_persists_document(
    api_client: AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0) for _ in texts]

    monkeypatch.setattr(ingestion_pipeline_module, "embed_texts", fake_embed_texts)

    pdf_bytes = make_pdf(tmp_path / "source.pdf").read_bytes()

    response = await api_client.post(
        "/ingest",
        files={"file": ("novatech_2024.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["company"] == "novatech"
    assert body["year"] == 2024
    assert (Path(settings.upload_dir) / "novatech_2024.pdf").is_file()


async def test_ingest_invalid_pdf_returns_400(
    api_client: AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))

    response = await api_client.post(
        "/ingest",
        files={"file": ("corrompu_2024.pdf", b"not a real pdf", "application/pdf")},
    )

    assert response.status_code == 400


# --- /query ------------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        for delta in self.deltas:
            yield delta


class FailingLLMClient(LLMClient):
    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        raise RuntimeError("quota Groq dépassé")
        yield  # pragma: no cover — needed so the method stays an async generator


async def test_query_streams_sources_then_deltas(
    api_client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        rag_pipeline_module, "get_llm_client", lambda: FakeLLMClient(["Bonjour", " le monde"])
    )

    document = Document(source_path="renault-2023.pdf", company="Renault", year=2023)
    db_session.add(document)
    await db_session.flush()
    db_session.add(
        Chunk(
            document_id=document.id,
            text="Le chiffre d'affaires de Renault en 2023 est de 42M€.",
            page_number=1,
            chunk_index=0,
            kind="text",
            embedding=unit_vector(0),
        )
    )
    await db_session.flush()

    async with api_client.stream(
        "POST", "/query", json={"query": "chiffre d'affaires", "company": "Renault"}
    ) as response:
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: sources" in body
    assert "Renault" in body
    assert "event: delta" in body
    assert "Bonjour" in body
    assert "event: done" in body


async def test_query_streams_error_event_on_llm_failure(
    api_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_pipeline_module, "get_llm_client", lambda: FailingLLMClient())

    async with api_client.stream(
        "POST", "/query", json={"query": "chiffre d'affaires"}
    ) as response:
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: error" in body
    assert "quota Groq" in body
