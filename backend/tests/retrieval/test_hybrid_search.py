"""Tests for HYBRID search (`hybrid_search`, RRF fusion, Phase 9).

The throughline is case **q4** (Phase 8): pure vector search ranked the numbers-table page ahead
of the identity page for the question "code APE / NAF?". These tests prove, on a controlled
scenario, that RRF fusion reverses that order — a chunk strong on lexical but weak on dense must
outrank a chunk strong on dense but absent from lexical.

Vectors are built as in `test_search.py` (`unit_vector`, predictable cosine similarity), text as
in `test_lexical_search.py` — combined here to control BOTH channels at once and verify the exact
RRF score (`RRF_K` imported from the module, not hardcoded).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EMBEDDING_DIM
from app.models import Chunk, Document
from app.retrieval import search as search_module
from app.retrieval.search import RRF_K, hybrid_search


def unit_vector(index: int, *, sign: float = 1.0) -> list[float]:
    """Unit vector (a single coordinate set to `sign`, the rest to 0) — predictable cosine
    similarity against the question vector (identical, orthogonal, or opposite)."""
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = sign
    return vector


async def make_chunk(
    session: AsyncSession,
    *,
    text: str,
    embedding: list[float],
    page_number: int = 1,
    company: str = "novatech",
    year: int = 2025,
) -> Chunk:
    document = Document(source_path=f"{company}-{year}.pdf", company=company, year=year)
    session.add(document)
    await session.flush()

    chunk = Chunk(
        document_id=document.id,
        text=text,
        page_number=page_number,
        chunk_index=0,
        kind="text",
        embedding=embedding,
    )
    session.add(chunk)
    await session.flush()
    return chunk


def mock_question_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """The question always embeds to `unit_vector(0)` — the reference point against which
    chunks are positioned (identical, orthogonal, or opposite) to control the dense rank."""

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)


async def test_hybrid_search_recovers_a_match_the_dense_channel_ranks_last(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case q4: a lexically strong but dense-weak chunk outranks a dense-strong chunk absent
    from lexical — the reversal that fixes retrieval (see Phase 8: pure vector search ranked the
    balance-sheet page ahead of the identity page)."""
    mock_question_embedding(monkeypatch)

    identity_page = await make_chunk(
        db_session,
        text="Forme juridique : SARL. Code APE : 4651Z.",
        embedding=unit_vector(1),  # orthogonal to the question → dense similarity = 0.0
        page_number=1,
    )
    balance_page = await make_chunk(
        db_session,
        text="Total du bilan : 457 150 euros.",
        embedding=unit_vector(0),  # identical to the question → dense similarity = 1.0 (rank 1)
        page_number=2,
    )

    results = await hybrid_search("Quel est le code APE / NAF ?", db_session)

    # Dense alone would have ranked balance_page first (similarity 1.0 vs 0.0, see Phase 8) —
    # fusion reverses the order because identity_page is the only lexical match (lexical rank 1).
    assert [r.chunk.id for r in results] == [identity_page.id, balance_page.id]
    assert results[0].similarity == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert results[1].similarity == pytest.approx(1 / (RRF_K + 1))


async def test_hybrid_search_returns_each_chunk_once(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk ranked first on BOTH channels must appear ONLY ONCE in the fused result, with a
    score equal to the sum of both contributions (not two separate entries)."""
    mock_question_embedding(monkeypatch)

    chunk = await make_chunk(
        db_session,
        text="Code APE : 4651Z",
        embedding=unit_vector(0),  # dense rank 1 AND lexical rank 1 (only chunk in the database)
    )

    results = await hybrid_search("code APE", db_session)

    assert [r.chunk.id for r in results] == [chunk.id]
    assert results[0].similarity == pytest.approx(2 / (RRF_K + 1))


async def test_hybrid_search_respects_k_after_fusion(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`k` applies AFTER fusion: with the default `fetch_k` (20) but `k=2`, a third candidate
    retrieved by both channels must not survive the final truncation."""
    mock_question_embedding(monkeypatch)

    best_dense = await make_chunk(
        db_session,
        text="Total du bilan : 457 150 euros.",  # no lexical match
        embedding=unit_vector(0),  # dense similarity = 1.0 → dense rank 1
    )
    mid_dense_but_lexical_match = await make_chunk(
        db_session,
        text="Code APE : 4651Z",  # the only chunk matching the lexical channel
        embedding=unit_vector(1),  # dense similarity = 0.0 → dense rank 2
    )
    worst_dense = await make_chunk(
        db_session,
        text="Immobilisations financières : 6 200 euros.",  # no lexical match
        embedding=unit_vector(0, sign=-1.0),  # dense similarity = -1.0 → dense rank 3 (worst)
    )

    results = await hybrid_search("code APE", db_session, k=2)

    # mid_dense_but_lexical_match wins despite a mediocre dense rank (sole lexical support);
    # worst_dense, last on both channels, is dropped by the truncation at k=2.
    assert [r.chunk.id for r in results] == [mid_dense_but_lexical_match.id, best_dense.id]
    assert worst_dense.id not in {r.chunk.id for r in results}
