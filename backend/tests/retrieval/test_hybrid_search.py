"""Tests de la recherche HYBRIDE (`hybrid_search`, fusion RRF, Phase 9).

Le fil rouge est le cas **q4** (Phase 8) : la recherche vectorielle pure classait la page de
tableaux chiffrés devant la page d'identité pour la question « code APE / NAF ? ». Ces tests
prouvent, sur un scénario contrôlé, que la fusion RRF inverse cet ordre — un chunk fort au
lexical mais faible au dense doit passer DEVANT un chunk fort au dense mais absent du lexical.

Les vecteurs sont construits comme dans `test_search.py` (`unit_vector`, similarité cosinus
prévisible), le texte comme dans `test_lexical_search.py` — combinés ici pour contrôler les DEUX
canaux à la fois et vérifier le score RRF exact (`RRF_K` importé du module, pas recopié en dur).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EMBEDDING_DIM
from app.models import Chunk, Document
from app.retrieval import search as search_module
from app.retrieval.search import RRF_K, hybrid_search


def unit_vector(index: int, *, sign: float = 1.0) -> list[float]:
    """Vecteur unitaire (une seule coordonnée à `sign`, le reste à 0) — similarité cosinus
    prévisible face au vecteur de la question (identique, orthogonal, ou opposé)."""
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
    """La question s'embarque toujours en `unit_vector(0)` — le point de référence face auquel
    on positionne les chunks (identique, orthogonal, ou opposé) pour contrôler le rang dense."""

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [unit_vector(0)]

    monkeypatch.setattr(search_module, "embed_texts", fake_embed_texts)


async def test_hybrid_search_recovers_a_match_the_dense_channel_ranks_last(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cas q4 : un chunk lexicalement fort mais dense-faible passe DEVANT un chunk dense-fort
    mais absent du lexical — le renversement qui répare le retrieval (cf. Phase 8 : la recherche
    vectorielle pure classait la page bilan devant la page identité)."""
    mock_question_embedding(monkeypatch)

    identity_page = await make_chunk(
        db_session,
        text="Forme juridique : SARL. Code APE : 4651Z.",
        embedding=unit_vector(1),  # orthogonal à la question → similarité dense = 0.0
        page_number=1,
    )
    balance_page = await make_chunk(
        db_session,
        text="Total du bilan : 457 150 euros.",
        embedding=unit_vector(0),  # identique à la question → similarité dense = 1.0 (rang 1)
        page_number=2,
    )

    results = await hybrid_search("Quel est le code APE / NAF ?", db_session)

    # Dense seul aurait classé balance_page en tête (similarité 1.0 vs 0.0, cf. Phase 8) — la
    # fusion inverse l'ordre car identity_page est seule à matcher lexicalement (rang 1 lexical).
    assert [r.chunk.id for r in results] == [identity_page.id, balance_page.id]
    assert results[0].similarity == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert results[1].similarity == pytest.approx(1 / (RRF_K + 1))


async def test_hybrid_search_returns_each_chunk_once(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un chunk présent en tête des DEUX canaux doit apparaître UNE SEULE fois dans le résultat
    fusionné, avec un score = somme des deux contributions (pas deux entrées séparées)."""
    mock_question_embedding(monkeypatch)

    chunk = await make_chunk(
        db_session,
        text="Code APE : 4651Z",
        embedding=unit_vector(0),  # rang 1 dense ET rang 1 lexical (seul chunk en base)
    )

    results = await hybrid_search("code APE", db_session)

    assert [r.chunk.id for r in results] == [chunk.id]
    assert results[0].similarity == pytest.approx(2 / (RRF_K + 1))


async def test_hybrid_search_respects_k_after_fusion(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`k` s'applique APRÈS fusion : avec `fetch_k` par défaut (20) mais `k=2`, un troisième
    candidat récupéré par les deux canaux ne doit pas survivre à la troncature finale."""
    mock_question_embedding(monkeypatch)

    best_dense = await make_chunk(
        db_session,
        text="Total du bilan : 457 150 euros.",  # aucun match lexical
        embedding=unit_vector(0),  # similarité dense = 1.0 → rang 1 dense
    )
    mid_dense_but_lexical_match = await make_chunk(
        db_session,
        text="Code APE : 4651Z",  # seul chunk à matcher le canal lexical
        embedding=unit_vector(1),  # similarité dense = 0.0 → rang 2 dense
    )
    worst_dense = await make_chunk(
        db_session,
        text="Immobilisations financières : 6 200 euros.",  # aucun match lexical
        embedding=unit_vector(0, sign=-1.0),  # similarité dense = -1.0 → rang 3 dense (pire)
    )

    results = await hybrid_search("code APE", db_session, k=2)

    # mid_dense_but_lexical_match gagne malgré un rang dense médiocre (seul soutien lexical) ;
    # worst_dense, dernier sur les deux canaux, est écarté par la troncature à k=2.
    assert [r.chunk.id for r in results] == [mid_dense_but_lexical_match.id, best_dense.id]
    assert worst_dense.id not in {r.chunk.id for r in results}