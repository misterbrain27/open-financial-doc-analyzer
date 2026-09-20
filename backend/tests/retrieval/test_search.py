from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EMBEDDING_DIM
from app.models import Chunk, Document
from app.retrieval import search as search_module
from app.retrieval.search import search


def unit_vector(index: int, *, sign: float = 1.0) -> list[float]:
    """Unit vector (a single coordinate set to `sign`, the rest to 0) — handy to get a
    predictable cosine similarity between two vectors (identical, orthogonal, opposite)."""
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = sign
    return vector


async def make_document_with_chunk(
    session: AsyncSession, *, embedding: list[float], company: str, year: int
) -> Chunk:
    document = Document(source_path=f"{company}-{year}.pdf", company=company, year=year)
    session.add(document)
    await session.flush()

    chunk = Chunk(
        document_id=document.id,
        text=f"extrait {company} {year}",
        page_number=1,
        chunk_index=0,
        kind="text",
        embedding=embedding,
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def test_search_returns_top_k_ordered_by_similarity(
    db_session: AsyncSession, monkeypatch
) -> None:
    query_vector = unit_vector(0)

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [query_vector]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)

    identical = await make_document_with_chunk(
        db_session, embedding=unit_vector(0), company="Renault", year=2023
    )
    orthogonal = await make_document_with_chunk(
        db_session, embedding=unit_vector(1), company="Renault", year=2023
    )
    await make_document_with_chunk(
        db_session, embedding=unit_vector(0, sign=-1.0), company="Renault", year=2023
    )

    results = await search("chiffre d'affaires", db_session, k=2)

    assert [r.chunk.id for r in results] == [identical.id, orthogonal.id]
    assert results[0].similarity == 1.0
    assert results[1].similarity == 0.0


async def test_search_filters_by_company_and_year(db_session: AsyncSession, monkeypatch) -> None:
    query_vector = unit_vector(0)

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [query_vector]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)

    renault_chunk = await make_document_with_chunk(
        db_session, embedding=unit_vector(0), company="Renault", year=2023
    )
    await make_document_with_chunk(
        db_session, embedding=unit_vector(0), company="TotalEnergies", year=2022
    )

    results = await search("chiffre d'affaires", db_session, company="Renault", year=2023)

    assert [r.chunk.id for r in results] == [renault_chunk.id]


async def test_search_company_filter_is_case_insensitive(
    db_session: AsyncSession, monkeypatch
) -> None:
    query_vector = unit_vector(0)

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [query_vector]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)

    renault_chunk = await make_document_with_chunk(
        db_session, embedding=unit_vector(0), company="Renault", year=2023
    )

    # The stored value is "Renault" ; the filter must match regardless of case.
    results = await search("chiffre d'affaires", db_session, company="RENAULT")

    assert [r.chunk.id for r in results] == [renault_chunk.id]
