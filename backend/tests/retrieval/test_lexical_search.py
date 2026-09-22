"""Tests de la recherche LEXICALE (`lexical_search`, Phase 9).

On valide le signal lexical isolément — le canal qui rattrape ce que le vectoriel manque :
- il **classe en tête** le chunk qui contient le terme exact (cas q4 : code APE, page 1) ;
- il matche en **OU**, donc trouve un chunk même quand tous les termes de la question n'y sont
  pas (« NAF » est absent du corpus — un ET strict ne remonterait rien) ;
- il respecte les mêmes **filtres** `company` / `year` que `search()`.

Ces tests tournent contre un vrai Postgres (fixture `db_session`) : la colonne générée `tsv` et
son `to_tsvector('french', …)` sont calculés par la base, pas simulés.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EMBEDDING_DIM
from app.models import Chunk, Document
from app.retrieval.search import lexical_search

# `lexical_search` n'utilise jamais l'embedding, mais la colonne n'est pas nullable et l'index
# HNSW (cosinus) refuse un vecteur nul. Un vecteur unitaire constant convient à toutes les lignes.
EMBEDDING = [1.0] + [0.0] * (EMBEDDING_DIM - 1)


async def make_chunk(
    session: AsyncSession,
    *,
    text: str,
    page_number: int = 1,
    company: str = "novatech",
    year: int = 2025,
) -> Chunk:
    """Insère un document + un chunk portant `text` (dont Postgres dérivera la colonne `tsv`)."""
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
    """Cas q4 : la question « code APE » remonte la page d'identité en tête.

    C'est précisément le cas que la recherche vectorielle rate (elle ramène la page de tableaux).
    Le chunk d'identité matche deux lexèmes (`code` + `ape`), celui du code de commerce un seul
    (`code`) → `ts_rank_cd` le classe devant ; la page de bilan ne matche rien et sort du résultat.
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

    # `balance_sheet` ne matche aucun terme → exclu par `@@` ; l'identité passe avant le commerce.
    assert [r.chunk.id for r in results] == [identity.id, commerce.id]
    assert balance_sheet.id not in {r.chunk.id for r in results}


async def test_lexical_search_uses_or_semantics(db_session: AsyncSession) -> None:
    """Verrouille la décision OU : un chunk est trouvé même s'il ne contient pas TOUS les termes.

    Le chunk porte `code` + `ape` mais **pas** `naf` (absent du corpus, comme dans le vrai
    document). Avec un ET strict (`plainto_tsquery` par défaut), la requête `code & ape & naf`
    ne matcherait rien. En OU, elle remonte bien le chunk — c'est ce qui répare q4.
    """
    chunk = await make_chunk(db_session, text="Code APE : 4651Z", page_number=1)

    results = await lexical_search("code APE NAF", db_session, k=5)

    assert [r.chunk.id for r in results] == [chunk.id]


async def test_lexical_search_filters_by_company(db_session: AsyncSession) -> None:
    """Le filtre `company` s'applique au lexical comme au vectoriel, insensible à la casse."""
    novatech = await make_chunk(
        db_session, text="Code APE : 4651Z", page_number=1, company="novatech"
    )
    await make_chunk(db_session, text="Code APE : 9999X", page_number=1, company="autre-sa")

    # Stocké « novatech » ; on interroge en majuscules → le filtre doit quand même matcher.
    results = await lexical_search("code APE", db_session, company="NOVATECH")

    assert [r.chunk.id for r in results] == [novatech.id]
