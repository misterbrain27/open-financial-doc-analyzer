from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `scripts/` is NOT a package (pyproject only packages `app*`): we add `backend/`
# to sys.path so `app.*` can be imported regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def iter_pdfs(target: Path) -> list[Path]:
    """The file itself, or all *.pdf files sorted from a directory."""
    if target.is_dir():
        return sorted(target.glob("*.pdf"))
    return [target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingère des PDF financiers ")
    parser.add_argument("path", type=Path, help="Fichier PDF ou dossier de PDF")
    args = parser.parse_args()

    paths = iter_pdfs(args.path)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
