"""Compares the 3 retrieval methods (vector, lexical, hybrid RRF) for a given question.

Debugging tool for Phase 9 (hybrid search): displays, side by side, what each method
returns — useful for understanding WHY the hybrid approach fixes (or doesn't) a given case,
without navigating the database blindly. Assumes an already-initialized database with
documents already ingested (`make local` + `ingest_cli.py`) — this script writes nothing, it
only inspects what already exists.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# `scripts/` is NOT a package (pyproject only packages `app*`): we add `backend/`
# to sys.path so `app.*` can be imported regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import async_session  # noqa: E402
from app.retrieval.search import SearchResult, hybrid_search, lexical_search, search  # noqa: E402


def format_row(rank: int, result: SearchResult) -> str:
    preview = result.chunk.text if len(result.chunk.text) <= 80 else result.chunk.text[:77] + "..."
    preview = preview.replace("\n", " ⏎ ")
    company = result.company or "?"
    year = result.year or "?"
    return (
        f"  {rank:>2}. score={result.similarity:+.4f} | p{result.chunk.page_number} "
        f"{company}/{year} | {preview}"
    )


def print_ranking(title: str, results: list[SearchResult]) -> None:
    print(f"\n── {title} ──")
    if not results:
        print("  (aucun résultat)")
        return
    for rank, result in enumerate(results, start=1):
        print(format_row(rank, result))


async def run(query: str, *, k: int, company: str | None, year: int | None) -> None:
    async with async_session() as session:
        vector_results = await search(query, session, k=k, company=company, year=year)
        lexical_results = await lexical_search(query, session, k=k, company=company, year=year)
        hybrid_results = await hybrid_search(query, session, k=k, company=company, year=year)

    print(f"Question : {query!r}  (k={k}, company={company!r}, year={year!r})")
    print_ranking("Vectoriel (cosinus)", vector_results)
    print_ranking("Lexical (ts_rank_cd, OU)", lexical_results)
    print_ranking("Hybride (RRF)", hybrid_results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vectoriel / lexical / hybride (RRF) pour une question (Phase 9, "
            "recherche hybride) — outil de mise au point du retrieval."
        )
    )
    parser.add_argument("query", help="Question à rechercher")
    parser.add_argument("-k", type=int, default=5, help="Nombre de résultats par méthode")
    parser.add_argument("--company", default=None, help="Filtre entreprise (optionnel)")
    parser.add_argument("--year", type=int, default=None, help="Filtre année (optionnel)")
    args = parser.parse_args()

    asyncio.run(run(args.query, k=args.k, company=args.company, year=args.year))


if __name__ == "__main__":
    main()
