from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# `scripts/` is NOT a package (pyproject only packages `app*`): we add `backend/`
# to sys.path so `app.*` can be imported regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import async_session, init_db  # noqa: E402
from app.ingestion.pipeline import ingest_pdf  # noqa: E402


def iter_pdfs(target: Path) -> list[Path]:
    """The file itself, or all *.pdf files sorted from a directory."""
    if target.is_dir():
        return sorted(target.glob("*.pdf"))
    return [target]


async def run(pdf_paths: list[Path], *, chunk_size: int, overlap: int) -> None:
    await init_db()
    async with async_session() as session:
        for pdf_path in pdf_paths:
            document = await ingest_pdf(pdf_path, session, chunk_size=chunk_size, overlap=overlap)
            print(f"{pdf_path.name} -> document id={document.id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingère des PDF financiers en base (Phase 3) : chunks + embeddings pgvector."
    )
    parser.add_argument("path", type=Path, help="Fichier PDF ou dossier de PDF")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Taille cible d'un chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Chevauchement entre chunks")
    args = parser.parse_args()

    asyncio.run(run(iter_pdfs(args.path), chunk_size=args.chunk_size, overlap=args.overlap))


if __name__ == "__main__":
    main()
