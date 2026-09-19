from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `scripts/` is NOT a package (pyproject only packages `app*`): we add `backend/`
# to sys.path so `app.*` can be imported regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.chunker import chunk_document, format_chunk_summary  # noqa: E402
from app.ingestion.loader import load_pdf  # noqa: E402


def iter_pdfs(target: Path) -> list[Path]:
    """The file itself, or all *.pdf files sorted from a directory."""
    if target.is_dir():
        return sorted(target.glob("*.pdf"))
    return [target]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspecte le chunking de PDF financiers (Phase 2)."
    )
    parser.add_argument("path", type=Path, help="Fichier PDF ou dossier de PDF")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Taille cible d'un chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Chevauchement entre chunks")
    parser.add_argument("--show", action="store_true", help="Affiche le contenu de chaque chunk")
    args = parser.parse_args()

    for pdf_path in iter_pdfs(args.path):
        doc = load_pdf(pdf_path)
        chunks = chunk_document(doc, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"{pdf_path.name} | {format_chunk_summary(chunks)}")
        if args.show:
            for c in chunks:
                preview = c.text if len(c.text) <= 100 else c.text[:97] + "..."
                preview = preview.replace("\n", " ⏎ ")  # multi-line tables → single line
                print(f"  [{c.chunk_index:>3}] p{c.page_number} {c.kind:<5} | {preview}")


if __name__ == "__main__":
    main()
