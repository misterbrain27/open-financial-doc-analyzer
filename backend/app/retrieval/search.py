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


def _or_tsquery(query: str):
    """Construit une `tsquery` française reliant les termes par OU (`|`).

    `plainto_tsquery` relie les lexèmes par ET (`&`), trop strict pour nos questions : « code
    APE / NAF » exigerait le lexème `naf`, **absent du corpus** → zéro match, et l'hybride n'y
    gagnerait rien (cf. Phase 8, q4). En OU, un chunk qui contient *au moins un* terme matche,
    et `ts_rank_cd` classe par densité/proximité (la page 1 a `code` + `ape` → remonte en tête).

    Technique 100 % SQL et sûre : `plainto_tsquery` fait déjà le nettoyage délicat (stop-words,
    stemming, ponctuation) ; on remplace juste ` & ` par ` | ` dans sa forme textuelle, puis on
    recaste en `tsquery`. (Construire un `to_tsquery` à la main serait fragile : un stop-word
    adjacent à un opérateur le fait planter.)
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
    """Recherche LEXICALE (full-text Postgres) — le pendant mot-clé de `search()`.

    Classe les chunks par `ts_rank_cd` sur la colonne générée `Chunk.tsv` (config `french`).
    Rattrape les lookups exacts (codes, identifiants) que le vectoriel manque — cf. Phase 8
    (q4 code APE, q2 capital social). Les deux signaux seront fusionnés par RRF (Phase 9, B).

    Même signature et même type de retour que `search()`, à une nuance près : pour un résultat
    lexical, `SearchResult.similarity` porte le score **`ts_rank_cd`** (non borné [0, +∞[), pas
    une similarité cosinus. L'ordre est ce qui compte — la fusion RRF ne regarde que les rangs.

    Args:
        query: Texte à rechercher.
        session: Session SQLAlchemy asynchrone.
        k: Nombre de résultats à renvoyer.
        company: Si fourni, restreint aux chunks du document de cette entreprise.
        year: Si fourni, restreint aux chunks du document de cette année.

    Returns:
        Liste de `SearchResult` triés par score lexical décroissant.
    """
    tsquery = _or_tsquery(query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("rank")
    stmt = select(Chunk, rank, Document.company, Document.year).join(
        Document, Chunk.document_id == Document.id
    )
    # `@@` : ne garder que les chunks dont le `tsv` matche la requête (sinon ts_rank_cd = 0
    # pour tout le reste et on remonterait du bruit).
    stmt = stmt.where(Chunk.tsv.bool_op("@@")(tsquery))
    if company is not None:
        # Même filtre insensible à la casse que `search()`.
        stmt = stmt.where(func.lower(Document.company) == company.lower())
    if year is not None:
        stmt = stmt.where(Document.year == year)
    stmt = stmt.order_by(rank.desc()).limit(k)
    result = await session.execute(stmt)
    return [
        SearchResult(chunk=chunk, similarity=score, company=doc_company, year=doc_year)
        for chunk, score, doc_company, doc_year in result
    ]


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


# Constante d'amortissement RRF (valeur usuelle) : adoucit l'écart entre rang 1 et rang 2,
# évite qu'un seul canal n'écrase l'autre à lui seul.
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
    """Recherche HYBRIDE : fusionne `search()` (dense) et `lexical_search()` par RRF.

    Récupère `fetch_k` candidats de chaque canal (plus que `k`, pour donner de la matière à la
    fusion), puis classe par score de *Reciprocal Rank Fusion* : chaque chunk cumule
    `1 / (RRF_K + rang)` pour chaque liste où il apparaît. On ne fusionne JAMAIS les scores bruts
    (cosinus vs `ts_rank_cd`, échelles incomparables) — seuls les RANGS comptent.

    Args:
        query: Texte à rechercher.
        session: Session SQLAlchemy asynchrone.
        k: Nombre de résultats à renvoyer, après fusion.
        company: Si fourni, restreint aux chunks du document de cette entreprise.
        year: Si fourni, restreint aux chunks du document de cette année.
        fetch_k: Nombre de candidats récupérés par canal, avant fusion.

    Returns:
        Liste de `SearchResult` triés par score RRF décroissant (`similarity` porte ce score RRF,
        non comparable aux similarités des canaux d'origine). En cas d'égalité de score, le tri
        stable de Python conserve l'ordre d'insertion dans `rrf_scores` : le canal dense (itéré en
        premier) l'emporte.
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
