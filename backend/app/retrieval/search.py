from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Text, cast, func, select
from sqlalchemy.dialects.postgresql import TSQUERY
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
    """Searches for text within chunks and returns results sorted by relevance.

    Args:
        query: Text to search for.
        session: Async SQLAlchemy session.
        k: Number of results to return.
        company: If provided, restricts to chunks from this company's document.
        year: If provided, restricts to chunks from this year's document.

    Returns:
        List of `SearchResult` sorted by relevance (descending score).
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


def _or_tsquery(query: str):
    """Builds a French `tsquery` joining terms with OR (`|`).

    `plainto_tsquery` joins lexemes with AND (`&`), too strict for our questions: "APE/NAF
    code" would require the lexeme `naf`, **absent from the corpus** → zero matches, and the
    hybrid search wouldn't gain anything from it (see Phase 8, q4). With OR, a chunk that
    contains *at least one* term matches, and `ts_rank_cd` ranks by density/proximity (page 1
    has `code` + `ape` → rises to the top).

    A 100% SQL, safe technique: `plainto_tsquery` already does the delicate cleanup (stop
    words, stemming, punctuation); we just replace ` & ` with ` | ` in its textual form, then
    recast it to `tsquery`. (Hand-building a `to_tsquery` would be fragile: a stop word next
    to an operator makes it crash.)
    """
    plain = func.plainto_tsquery("french", query)
    return cast(func.replace(cast(plain, Text), " & ", " | "), TSQUERY)


async def lexical_search(
    query: str,
    session: AsyncSession,
    k: int = 5,
    *,
    company: str | None = None,
    year: int | None = None,
) -> list[SearchResult]:
    """LEXICAL search (Postgres full-text) — the keyword counterpart of `search()`.

    Ranks chunks by `ts_rank_cd` on the generated `Chunk.tsv` column (`french` config).
    Catches exact lookups (codes, identifiers) that the vector search misses — see Phase 8
    (q4 APE code, q2 share capital). Both signals get merged via RRF (Phase 9, B).

    Same signature and same return type as `search()`, with one nuance: for a lexical
    result, `SearchResult.similarity` carries the **`ts_rank_cd`** score (unbounded [0, +∞[),
    not a cosine similarity. What matters is the ORDER — RRF fusion only looks at ranks.

    Args:
        query: Text to search for.
        session: Async SQLAlchemy session.
        k: Number of results to return.
        company: If provided, restricts to chunks from this company's document.
        year: If provided, restricts to chunks from this year's document.

    Returns:
        List of `SearchResult` sorted by descending lexical score.
    """
    tsquery = _or_tsquery(query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("rank")
    stmt = select(Chunk, rank, Document.company, Document.year).join(
        Document, Chunk.document_id == Document.id
    )
    # `@@`: keep only chunks whose `tsv` matches the query (otherwise ts_rank_cd = 0
    # for everything else and we'd surface noise).
    stmt = stmt.where(Chunk.tsv.bool_op("@@")(tsquery))
    if company is not None:
        # Same case-insensitive filter as `search()`.
        stmt = stmt.where(func.lower(Document.company) == company.lower())
    if year is not None:
        stmt = stmt.where(Document.year == year)
    stmt = stmt.order_by(rank.desc()).limit(k)
    result = await session.execute(stmt)
    return [
        SearchResult(chunk=chunk, similarity=score, company=doc_company, year=doc_year)
        for chunk, score, doc_company, doc_year in result
    ]


# RRF damping constant (usual value): softens the gap between rank 1 and rank 2,
# prevents a single channel from overpowering the other on its own.
RRF_K = 60


async def hybrid_search(
    query: str,
    session: AsyncSession,
    k: int = 5,
    *,
    company: str | None = None,
    year: int | None = None,
    fetch_k: int = 20,
) -> list[SearchResult]:
    """HYBRID search: merges `search()` (dense) and `lexical_search()` via RRF.

    Fetches `fetch_k` candidates from each channel (more than `k`, to give the fusion enough
    material to work with), then ranks by *Reciprocal Rank Fusion* score: each chunk
    accumulates `1 / (RRF_K + rank)` for every list it appears in. Raw scores are NEVER
    merged (cosine vs `ts_rank_cd`, incomparable scales) — only RANKS matter.

    Args:
        query: Text to search for.
        session: Async SQLAlchemy session.
        k: Number of results to return, after fusion.
        company: If provided, restricts to chunks from this company's document.
        year: If provided, restricts to chunks from this year's document.
        fetch_k: Number of candidates fetched per channel, before fusion.

    Returns:
        List of `SearchResult` sorted by descending RRF score (`similarity` carries this RRF
        score, not comparable to the original channels' similarities). In case of a score
        tie, Python's stable sort keeps the insertion order in `rrf_scores`: the dense
        channel (iterated first) wins.
    """
    vector_results = await search(query, session, k=fetch_k, company=company, year=year)
    lexical_results = await lexical_search(query, session, k=fetch_k, company=company, year=year)

    rrf_scores: dict[int, float] = {}
    result_by_chunk_id: dict[int, SearchResult] = {}
    for results in (vector_results, lexical_results):
        for rank, result in enumerate(results, start=1):
            rrf_scores[result.chunk.id] = rrf_scores.get(result.chunk.id, 0.0) + 1 / (RRF_K + rank)
            result_by_chunk_id.setdefault(result.chunk.id, result)

    ranked_ids = sorted(rrf_scores, key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
    return [
        SearchResult(
            chunk=result_by_chunk_id[chunk_id].chunk,
            similarity=rrf_scores[chunk_id],
            company=result_by_chunk_id[chunk_id].company,
            year=result_by_chunk_id[chunk_id].year,
        )
        for chunk_id in ranked_ids[:k]
    ]
