"""Pipeline RAG (Phase 5) : recherche des extraits pertinents (Phase 4), construit un prompt
augmenté et streame la réponse du LLM (Phase 5 — `rag/llm.py`) avec citations des sources.

Anti-hallucination / grounding : le prompt système contraint le modèle à ne répondre qu'à
partir des extraits fournis et à citer leur numéro. Les métadonnées de citation (`Source`)
viennent directement de la recherche (Phase 4), pas d'une extraction depuis le texte généré —
elles restent donc fiables même si le modèle se trompe dans ses citations inline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.llm import LLMClient, get_llm_client
from app.retrieval.search import SearchResult, hybrid_search

SYSTEM_PROMPT = (
    "Tu es un assistant d'analyse de documents financiers. Réponds uniquement à partir des "
    "extraits de sources fournis ci-dessous, en français. Si les extraits ne permettent pas "
    "de répondre, dis-le explicitement plutôt que d'inventer une réponse. Cite tes "
    "affirmations à l'aide du numéro de la source entre crochets, par exemple [1]."
)


@dataclass
class Source:
    index: int
    company: str | None
    year: int | None
    page_number: int
    text: str


@dataclass
class RagAnswer:
    sources: list[Source]
    stream: AsyncIterator[str]


def _build_sources(results: list[SearchResult]) -> list[Source]:
    return [
        Source(
            index=i,
            company=result.company,
            year=result.year,
            page_number=result.chunk.page_number,
            text=result.chunk.text,
        )
        for i, result in enumerate(results, start=1)
    ]


def _build_context(sources: list[Source]) -> str:
    blocks = [
        f"[{source.index}] ({source.company or '?'}, {source.year or '?'}, "
        f"p.{source.page_number})\n{source.text}"
        for source in sources
    ]
    return "\n\n".join(blocks)


async def answer_query(
    query: str,
    session: AsyncSession,
    *,
    k: int = 5,
    company: str | None = None,
    year: int | None = None,
    llm_client: LLMClient | None = None,
) -> RagAnswer:
    """Recherche les extraits pertinents puis prépare une réponse groundée avec citations.

    Le texte de la réponse est streamé (`RagAnswer.stream`) ; les sources, elles, sont déjà
    connues à cet instant (issues de la recherche, pas de la génération).
    """
    results = await hybrid_search(query, session, k=k, company=company, year=year)
    sources = _build_sources(results)
    client = llm_client or get_llm_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extraits de sources :\n\n{_build_context(sources)}\n\nQuestion : {query}",
        },
    ]

    return RagAnswer(sources=sources, stream=client.stream_chat(messages))
