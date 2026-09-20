"""RAG pipeline (Phase 5): retrieves relevant excerpts (Phase 4), builds an augmented prompt
and streams the LLM's response (Phase 5 — `rag/llm.py`) with source citations.

Anti-hallucination / grounding: the system prompt constrains the model to answer only from
the provided excerpts and to cite their number. The citation metadata (`Source`) comes
directly from the search (Phase 4), not from an extraction of the generated text — so it
stays reliable even if the model gets its inline citations wrong.
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
    """Searches for relevant excerpts then prepares a grounded answer with citations.

    The answer text is streamed (`RagAnswer.stream`); the sources, however, are already
    known at this point (they come from the search, not from generation).
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
