"""Tests for LEXICAL search (`lexical_search`, Phase 9).

Validates the lexical signal in isolation — the channel that catches what the vector search
misses:
- it **ranks first** the chunk containing the exact term (case q4: APE code, page 1);
- it matches on **OR**, so it finds a chunk even when not all the question's terms are in it
  ("NAF" is absent from the corpus — a strict AND would return nothing);
- it respects the same `company` / `year` **filters** as `search()`.

These tests run against a real Postgres (`db_session` fixture): the generated `tsv` column and
its `to_tsvector('french', …)` are computed by the database, not simulated.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EMBEDDING_DIM
from app.models import Chunk, Document
from app.retrieval.search import lexical_search

# `lexical_search` never uses the embedding, but the column isn't nullable and the HNSW
# (cosine) index rejects a zero vector. A constant unit vector works for every row.
EMBEDDING = [1.0] + [0.0] * (EMBEDDING_DIM - 1)


async def make_chunk(
    session: AsyncSession,
    *,
    text: str,
    page_number: int = 1,
    company: str = "novatech",
    year: int = 2025,
) -> Chunk:
    """Inserts a document + a chunk carrying `text` (Postgres derives the `tsv` column from it)."""
    document = Document(source_path=f"{company}-{year}.pdf", company=company, year=year)
    session.add(document)
    await session.flush()

    chunk = Chunk(
        document_id=document.id,
        text=text,
        page_number=page_number,
        chunk_index=0,
        kind="text",
        embedding=EMBEDDING,
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def test_lexical_search_ranks_exact_term_match_first(db_session: AsyncSession) -> None:
    """Case q4: the question "code APE" ranks the identity page first.

    This is exactly the case that vector search misses (it returns the tables page instead).
    The identity chunk matches two lexemes (`code` + `ape`), the commercial-code chunk only one
    (`code`) → `ts_rank_cd` ranks it ahead; the balance-sheet page matches nothing and drops out.
    """
    identity = await make_chunk(
        db_session,
        text="Forme juridique : SARL. Code APE : 4651Z. Capital social : 50 000 euros.",
        page_number=1,
    )
    balance_sheet = await make_chunk(
        db_session,
        text="Total du bilan : 457 150 euros. Immobilisations financières : 6 200.",
        page_number=2,
    )
    commerce = await make_chunk(
        db_session,
        text="En application du code de commerce, les comptes annuels sont établis.",
        page_number=4,
    )

    results = await lexical_search("Quel est le code APE / NAF ?", db_session, k=5)

    # `balance_sheet` matches no term → excluded by `@@`; the identity chunk beats the commerce one.
    assert [r.chunk.id for r in results] == [identity.id, commerce.id]
    assert balance_sheet.id not in {r.chunk.id for r in results}


async def test_lexical_search_uses_or_semantics(db_session: AsyncSession) -> None:
    """Locks in the OR decision: a chunk is found even if it doesn't contain ALL the terms.

    The chunk carries `code` + `ape` but **not** `naf` (absent from the corpus, as in the real
    document). With a strict AND (`plainto_tsquery`'s default), the query `code & ape & naf`
    would match nothing. With OR, it does find the chunk — this is what fixes q4.
    """
    chunk = await make_chunk(db_session, text="Code APE : 4651Z", page_number=1)

    results = await lexical_search("code APE NAF", db_session, k=5)

    assert [r.chunk.id for r in results] == [chunk.id]


async def test_lexical_search_filters_by_company(db_session: AsyncSession) -> None:
    """The `company` filter applies to lexical search just like vector search, case-insensitive."""
    novatech = await make_chunk(
        db_session, text="Code APE : 4651Z", page_number=1, company="novatech"
    )
    await make_chunk(db_session, text="Code APE : 9999X", page_number=1, company="autre-sa")

    # Stored as "novatech" ; querying in uppercase must still match.
    results = await lexical_search("code APE", db_session, company="NOVATECH")

    assert [r.chunk.id for r in results] == [novatech.id]
