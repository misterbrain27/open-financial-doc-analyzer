from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import embed_texts
from app.models import Chunk, Document


@dataclass
class SearchResult:
    chunk: Chunk
    similarity: float
    company: str | None
    year: int | None


async def search(
    query: str,
    session: AsyncSession,
    k: int = 5,
    *,
    company: str | None = None,
    year: int | None = None,
) -> list[SearchResult]:
    """Recherche un texte dans les chunks et renvoie les résultats triés par pertinence.

    Args:
        query: Texte à rechercher.
        session: Session SQLAlchemy asynchrone.
        k: Nombre de résultats à renvoyer.
        company: Si fourni, restreint aux chunks du document de cette entreprise.
        year: Si fourni, restreint aux chunks du document de cette année.

    Returns:
        Liste de `SearchResult` triés par pertinence (score décroissant).
    """
    vectors = await embed_texts([query])
    query_vector = vectors[0]
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = select(Chunk, distance, Document.company, Document.year).join(
        Document, Chunk.document_id == Document.id
    )
    if company is not None:
        # Case-insensitive filter: the user might type "Novatech" while the stored value
        # is "novatech". Compare both sides lowercased.
        stmt = stmt.where(func.lower(Document.company) == company.lower())
    if year is not None:
        stmt = stmt.where(Document.year == year)
    stmt = stmt.order_by(distance).limit(k)
    result = await session.execute(stmt)
    return [
        SearchResult(chunk=chunk, similarity=1 - distance, company=doc_company, year=doc_year)
        for chunk, distance, doc_company, doc_year in result
    ]
