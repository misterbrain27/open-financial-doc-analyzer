"""Phase 2 — Chunking of text into overlapping fragments.

Turns a document's cleaned text (Phase 1) into a list of `Chunk` ready to be
embedded as vectors (Phase 3). Three levels:
- `split_text`: pure function, splits **one** string into windows anchored at boundaries
  (targeting `chunk_size` characters, but only cutting at whitespace — never in the
  middle of a word or a number, a finance concern) with ~`overlap` characters of overlap;
- `serialize_table`: flattens a table grid into indexable text (`row | by | row`);
- `chunk_document`: walks a `LoadedDocument` and wraps each text fragment **and
  each table** in a `Chunk` carrying its metadata (source, company, year, page,
  global index). A table is **never** split (it would break its structure): 1 table = 1 chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imports under TYPE_CHECKING only: the `LoadedDocument` / `Table` annotations are
    # never evaluated at runtime (see `from __future__ import annotations`), so the
    # chunker stays independent of pdfplumber (imported by loader.py).
    from app.ingestion.loader import LoadedDocument, Table


def split_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Splits `text` into fragments of ≤ `chunk_size` characters, overlapping by ~`overlap`.

    The cut only happens at whitespace: no word or number is split (except a
    "giant token" longer than `chunk_size`, cut cleanly for lack of a better option). Two
    consecutive chunks share ~`overlap` characters, preserving boundary context.

    Raises `ValueError` if `overlap >= chunk_size`: without progress margin, the window
    would never advance (infinite loop).
    """
    # Parameter guard: progress relies on `chunk_size > overlap` (see fallback below).
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) doit être strictement inférieur à chunk_size ({chunk_size})"
        )

    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size  # "target" end of the window
        if end < len(text):  # (if end >= len, we take the whole rest as-is)
            # We move `end` back to the last whitespace in the window → never cut a word.
            end = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            # Fallback covering TWO cases at once:
            #  - no whitespace found (giant token) → rfind returns -1;
            #  - whitespace too early: `end <= start + overlap` would make the next
            #    `start = end - overlap` regress (data loss / loop).
            # In both cases we cut cleanly at `start + chunk_size`. Since chunk_size > overlap,
            # the next start = start + (chunk_size - overlap) > start → progress guaranteed.
            if end <= start + overlap:
                end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # we never add an empty chunk
            chunks.append(chunk)
        if end >= len(text):  # last piece emitted
            break
        start = end - overlap  # the next window steps back by `overlap` → overlap
    return chunks


@dataclass
class Chunk:
    """Fragment ready to embed: the text + the metadata inherited from the document.

    This metadata is used for retrieval (company/year filters, Phase 4) and for
    citations (source + page, Phase 5). `chunk_index` is **global to the document**.
    """

    text: str
    source_path: str
    company: str | None
    year: int | None
    page_number: int
    chunk_index: int
    kind: str = "text"  # "text" | "table"


def serialize_table(table: Table) -> str:
    """Flattens a table grid into indexable text: cells separated by ` | `, rows
    by a newline. `None` (absent) cells become empty.

    We deliberately keep a simple, readable formatting (`Item | 2022 | 2021`) rather
    than Markdown rendering: the goal is text the embedding can capture, not a display.
    """
    return "\n".join(" | ".join(cell or "" for cell in row) for row in table)


def chunk_document(
    doc: LoadedDocument, *, chunk_size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Splits a loaded document into `Chunk` (text **and** tables), metadata propagated.

    For each page, we first emit the text chunks (`split_text` on `page.text`), then
    one chunk per table (`serialize_table`, `kind="table"`). `chunk_index` runs across the
    whole document (0-indexed, increasing through pages and tables); nothing that produces
    emptiness adds a chunk or leaves a gap in the numbering.

    A table is **never split**: fragmenting it would destroy its row/column structure.
    """
    chunks: list[Chunk] = []
    for page in doc.pages:
        # 1) The page's narrative text → windows with overlap.
        for fragment in split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                Chunk(
                    text=fragment,
                    source_path=doc.source_path,
                    company=doc.company,
                    year=doc.year,
                    page_number=page.page_number,
                    # global index = current position in the list (before appending) → 0,1,2,…
                    chunk_index=len(chunks),
                    kind="text",
                )
            )
        # 2) Each table on the page → a single chunk (not split).
        for table in page.tables:
            serialized = serialize_table(table)
            if serialized.strip():  # we skip a table that became empty
                chunks.append(
                    Chunk(
                        text=serialized,
                        source_path=doc.source_path,
                        company=doc.company,
                        year=doc.year,
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                        kind="table",
                    )
                )
    return chunks


def format_chunk_summary(chunks: list[Chunk]) -> str:
    """Readable one-line summary: chunk count, text/table split, lengths.

    Pure function (no I/O) → testable without capturing stdout; the inspection CLI uses it.
    """
    n_text = sum(1 for c in chunks if c.kind == "text")
    n_table = sum(1 for c in chunks if c.kind == "table")
    lengths = [len(c.text) for c in chunks]
    moyenne = sum(lengths) // len(lengths) if lengths else 0
    maximum = max(lengths) if lengths else 0
    return (
        f"{len(chunks)} chunks ({n_text} texte, {n_table} tableaux) "
        f"| long. moy={moyenne} max={maximum} car."
    )
