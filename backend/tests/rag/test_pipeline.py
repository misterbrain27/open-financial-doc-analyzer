from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EMBEDDING_DIM
from app.models import Chunk, Document
from app.rag import pipeline as pipeline_module
from app.rag.llm import LLMClient
from app.rag.pipeline import answer_query
from app.retrieval import search as search_module


class FakeLLMClient(LLMClient):
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.received_messages: list[dict[str, str]] | None = None

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.received_messages = messages
        for delta in self.deltas:
            yield delta


def unit_vector(index: int) -> list[float]:
    """Unit vector (a single coordinate set to 1, the rest to 0): gives a cosine similarity
    of 1.0 with itself — avoids the zero vector, whose cosine is undefined."""
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


async def make_document_with_chunk(
    session: AsyncSession, *, company: str, year: int, page_number: int = 1
) -> Chunk:
    document = Document(source_path=f"{company}-{year}.pdf", company=company, year=year)
    session.add(document)
    await session.flush()

    chunk = Chunk(
        document_id=document.id,
        text=f"Le chiffre d'affaires de {company} en {year} est de 42M€.",
        page_number=page_number,
        chunk_index=0,
        kind="text",
        embedding=unit_vector(0),
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def test_answer_query_builds_sources_from_search_results(
    db_session: AsyncSession, monkeypatch
) -> None:
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)

    await make_document_with_chunk(db_session, company="Renault", year=2023, page_number=12)

    fake_client = FakeLLMClient(["Le chiffre ", "d'affaires ", "est de 42M€ [1]."])

    result = await answer_query("chiffre d'affaires", db_session, llm_client=fake_client)

    assert len(result.sources) == 1
    assert result.sources[0] == pipeline_module.Source(
        index=1,
        company="Renault",
        year=2023,
        page_number=12,
        text="Le chiffre d'affaires de Renault en 2023 est de 42M€.",
    )


async def test_answer_query_streams_llm_output(db_session: AsyncSession, monkeypatch) -> None:
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)

    await make_document_with_chunk(db_session, company="Renault", year=2023)

    fake_client = FakeLLMClient(["Le chiffre ", "d'affaires ", "est de 42M€ [1]."])

    result = await answer_query("chiffre d'affaires", db_session, llm_client=fake_client)
    text = "".join([delta async for delta in result.stream])

    assert text == "Le chiffre d'affaires est de 42M€ [1]."


async def test_answer_query_grounds_prompt_with_context_and_system_message(
    db_session: AsyncSession, monkeypatch
) -> None:
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)

    await make_document_with_chunk(db_session, company="Renault", year=2023, page_number=12)

    fake_client = FakeLLMClient(["réponse"])

    result = await answer_query("chiffre d'affaires", db_session, llm_client=fake_client)
    async for _ in result.stream:
        pass

    assert fake_client.received_messages is not None
    system_message, user_message = fake_client.received_messages
    assert system_message == {"role": "system", "content": pipeline_module.SYSTEM_PROMPT}
    assert "[1]" in user_message["content"]
    assert "Renault" in user_message["content"]
    assert "chiffre d'affaires" in user_message["content"]
