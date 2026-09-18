"""Phase 1 — Loading and extraction of financial PDF documents.

Turns a PDF (annual report / AMF filing) into cleaned text + tables + metadata,
ready for chunking (Phase 2). We keep pure, testable functions:
- `clean_text`: normalization of the extracted text,
- `clean_table`: cleanup of extracted tables (cells, empty rows),
- `extract_metadata_from_filename`: derivation (company, year),
- `format_summary`: readable one-line summary of a loaded document,
- `load_pdf`: page-by-page orchestration via pdfplumber.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# A table extracted by pdfplumber: rows of cells (text or None).
Table = list[list[str | None]]


@dataclass
# This is a decorator that automatically generates the __init__, __repr__, __eq__, etc.
# methods for you. Without it, you'd have to write all of that by hand.
class PageContent:
    """Content of a page: number (1-indexed), cleaned text, tables."""

    page_number: int
    text: str
    tables: list[Table] = field(default_factory=list)


@dataclass
class LoadedDocument:
    """Loaded document: metadata + pages."""

    source_path: str
    company: str | None
    year: int | None
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        """Text of the entire document (non-empty pages), separated by newlines."""
        return "\n\n".join(page.text for page in self.pages if page.text)


_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# Table cells: `\s+` also includes newlines (\n) inside a cell,
# which `_WS_RE` (limited to spaces/tabs) would not.
_CELL_WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Normalizes the extracted text: multiple spaces/tabs collapsed, lines
    trimmed, excess newlines capped at a maximum of two."""
    if not raw:
        return ""
    lines = [_WS_RE.sub(" ", line).strip() for line in raw.splitlines()]
    text = "\n".join(lines)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def extract_metadata_from_filename(path: Path) -> tuple[str | None, int | None]:
    """Derives (company, year) from the filename.

    Recommended convention: `company_year.pdf` (e.g. `totalenergies_2023.pdf`).
    Stays robust if the pattern isn't followed.
    """
    stem = path.stem
    year_match = _YEAR_RE.search(stem)
    year = int(year_match.group()) if year_match else None

    company_raw = stem
    if year_match:
        company_raw = stem[: year_match.start()] + stem[year_match.end() :]
    company = company_raw.replace("_", " ").replace("-", " ").strip()
    return (company or None), year


def _clean_cell(cell: str | None) -> str | None:
    """Normalizes a table cell.

    `None` (cell absent from the pdfplumber grid) is kept as-is — it is distinguished
    from an empty cell `""`. Otherwise, internal whitespace (spaces, tabs, newlines)
    is collapsed to a single space, then the cell is trimmed.
    """
    if cell is None:
        return None
    return _CELL_WS_RE.sub(" ", cell).strip()


def clean_table(table: Table) -> Table:
    """Cleans a raw table extracted by pdfplumber.

    - each cell goes through `_clean_cell` (`None` distinguished from `""`);
    - rows that are **entirely empty** (all cells `None` or `""`) are removed.

    Does NOT fix merged columns or fragmentation: these are *detection* problems
    solved at the `table_settings` level, not by cleanup.
    """
    cleaned: Table = []
    for row in table:
        cleaned_row = [_clean_cell(cell) for cell in row]
        # `any(...)` is true as soon as one cell is a non-empty string (None and "" are falsy).
        if any(cell for cell in cleaned_row):
            cleaned.append(cleaned_row)
    return cleaned


def format_summary(doc: LoadedDocument) -> str:
    """Readable one-line summary: source, company, year, page count, table count, char count."""
    # source_path is a `str` -> we pass it back through `Path` to get `.name` (filename).
    name = Path(doc.source_path).name
    nb_pages = len(doc.pages)
    nb_tables = sum(len(page.tables) for page in doc.pages)
    nb_chars = len(doc.full_text)
    return (
        f"{name} | entreprise={doc.company or '?'} | année={doc.year or '?'} "
        f"| {nb_pages} pages | {nb_tables} tableaux | {nb_chars} car."
    )


def load_pdf(path: str | Path, *, extract_tables: bool = True) -> LoadedDocument:
    """Loads a PDF and returns a `LoadedDocument` (cleaned text + tables + metadata).

    Raises `FileNotFoundError` if the file doesn't exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {path}")

    company, year = extract_metadata_from_filename(path)
    pages: list[PageContent] = []

    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            raw_tables = page.extract_tables() if extract_tables else []
            # Cleans each table, then discards those that became empty. The walrus `:=` avoids
            # calling `clean_table` twice (once to filter, once for the value).
            tables = [cleaned for t in raw_tables if (cleaned := clean_table(t))]
            pages.append(PageContent(page_number=index, text=text, tables=tables))

    return LoadedDocument(source_path=str(path), company=company, year=year, pages=pages)
