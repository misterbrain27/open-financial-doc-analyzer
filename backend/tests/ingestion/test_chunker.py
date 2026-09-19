"""Tests for chunking (Phase 2).

`split_text` is a pure function: we test it directly with strings, without any PDF.
We favor **properties** (sizes, overlap, integrity) over fragile exact-output
assertions — except for targeted regressions.
"""

from __future__ import annotations

import pytest

from app.ingestion import chunker, loader

# Multi-sentence text with **distinct** words: any content loss is then immediately
# visible (identical characters would instead mask it).
SAMPLE = (
    "Le chiffre d'affaires progresse fortement cette annee dans toutes les regions du groupe "
    "tandis que le resultat net atteint un niveau record grace a la maitrise des couts fixes."
)


# --- Trivial cases ---------------------------------------------------------------------------


def test_split_text_short_returns_single_chunk():
    # Text shorter than chunk_size → a single chunk = the whole text.
    assert chunker.split_text("Bonjour le monde", chunk_size=1000, overlap=150) == [
        "Bonjour le monde"
    ]


def test_split_text_empty_returns_empty_list():
    assert chunker.split_text("", chunk_size=100, overlap=10) == []


def test_split_text_whitespace_only_returns_empty_list():
    assert chunker.split_text("   \n\t  ", chunk_size=100, overlap=10) == []


# --- Parameter guard (pitfall 2) -------------------------------------------------------------


def test_split_text_overlap_ge_chunk_size_raises():
    with pytest.raises(ValueError):
        chunker.split_text("a b c d e f g", chunk_size=10, overlap=10)


# --- Property: sizes --------------------------------------------------------------------------


def test_split_text_chunks_never_exceed_chunk_size():
    chunks = chunker.split_text(SAMPLE, chunk_size=40, overlap=10)
    assert len(chunks) >= 2  # the text is indeed split into multiple pieces
    assert all(len(c) <= 40 for c in chunks)


# --- Property: overlap --------------------------------------------------------------------


def test_split_text_consecutive_chunks_overlap():
    chunks = chunker.split_text(SAMPLE, chunk_size=40, overlap=15)
    assert len(chunks) >= 2
    # Two consecutive chunks must share at least one whole word (overlap zone).
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert set(a.split()) & set(b.split()), f"aucun chevauchement entre {a!r} et {b!r}"


# --- Property: integrity ------------------------------------------------------------------


def test_split_text_no_word_is_lost_or_split():
    # Strong property of this splitter: every word from the source is found WHOLE in at
    # least one chunk (it never cuts a word in half). Catches both loss and splitting.
    chunks = chunker.split_text(SAMPLE, chunk_size=40, overlap=10)
    for word in SAMPLE.split():
        assert any(word in c for c in chunks), f"mot perdu ou scinde : {word!r}"


# --- Edge cases: giant token + regression of pitfall 1 --------------------------------------


def test_split_text_giant_token_is_hard_split_without_looping():
    # A "word" with no whitespace at all, longer than chunk_size → hard cuts, but it
    # terminates and no chunk exceeds the size. (If progress weren't guaranteed, this would loop.)
    chunks = chunker.split_text("A" * 250, chunk_size=100, overlap=20)
    assert chunks
    assert all(set(c) == {"A"} for c in chunks)
    assert all(len(c) <= 100 for c in chunks)


def test_split_text_early_boundary_does_not_drop_content():
    # Regression of pitfall 1: an EARLY whitespace followed by a long run with no whitespace
    # used to push `start` below zero → data loss. Distinct characters so any gap is visible.
    run = "0123456789ABCDEFGHIJKLMNOPQRST"  # 30 characters, all different
    chunks = chunker.split_text("ok " + run, chunk_size=20, overlap=8)
    joined = "".join(chunks)
    for ch in run:
        assert ch in joined, f"caractere perdu : {ch!r}"


# --- chunk_document: hand-built LoadedDocument (no PDF) -------------------------


def test_chunk_document_propagates_metadata_and_indexes_globally():
    doc = loader.LoadedDocument(
        source_path="/data/raw/acme_2022.pdf",
        company="acme",
        year=2022,
        pages=[
            loader.PageContent(page_number=1, text="alpha beta gamma delta epsilon zeta eta"),
            loader.PageContent(page_number=2, text="Court."),  # < chunk_size → a single chunk
        ],
    )
    chunks = chunker.chunk_document(doc, chunk_size=15, overlap=3)

    assert len(chunks) >= 3  # page 1 split into multiple pieces + page 2 in a single one
    # inherited metadata, identical across all chunks
    assert all(c.source_path == "/data/raw/acme_2022.pdf" for c in chunks)
    assert all(c.company == "acme" and c.year == 2022 for c in chunks)
    assert all(c.kind == "text" for c in chunks)
    # sequential global index 0..n-1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # document order: non-decreasing page numbers, page 2 comes last
    page_numbers = [c.page_number for c in chunks]
    assert page_numbers == sorted(page_numbers)
    assert page_numbers[0] == 1 and page_numbers[-1] == 2
    # page 2 ("Court.") yields exactly one chunk, text intact
    page2 = [c for c in chunks if c.page_number == 2]
    assert len(page2) == 1
    assert page2[0].text == "Court."


def test_chunk_document_skips_empty_pages_without_index_gap():
    doc = loader.LoadedDocument(
        source_path="x.pdf",
        company=None,
        year=None,
        pages=[
            loader.PageContent(page_number=1, text="premier"),
            loader.PageContent(page_number=2, text="   \n  "),  # empty after strip → 0 chunks
            loader.PageContent(page_number=3, text="troisieme"),
        ],
    )
    chunks = chunker.chunk_document(doc, chunk_size=100, overlap=10)

    # page 2 is skipped, but numbering stays continuous (no 0,2 gap)
    assert [c.page_number for c in chunks] == [1, 3]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.text for c in chunks] == ["premier", "troisieme"]


def test_chunk_document_no_pages_returns_empty():
    doc = loader.LoadedDocument(source_path="x.pdf", company=None, year=None, pages=[])
    assert chunker.chunk_document(doc) == []


# --- serialize_table + tables in chunk_document -----------------------------------------


def test_serialize_table_joins_cells_and_rows():
    table = [["Poste", "2022", "2021"], ["Chiffre d'affaires", "1000", "900"]]
    assert chunker.serialize_table(table) == (
        "Poste | 2022 | 2021\nChiffre d'affaires | 1000 | 900"
    )


def test_serialize_table_none_cells_become_empty():
    # None (cell absent from the grid) → empty string, the column structure is preserved.
    assert chunker.serialize_table([["Total", None, "200"]]) == "Total |  | 200"


def test_chunk_document_emits_one_chunk_per_table_with_metadata():
    doc = loader.LoadedDocument(
        source_path="/data/raw/acme_2022.pdf",
        company="acme",
        year=2022,
        pages=[
            loader.PageContent(
                page_number=1,
                text="Bilan simplifie de l'exercice.",
                tables=[[["Poste", "2022"], ["Chiffre d'affaires", "1000"]]],
            ),
        ],
    )
    chunks = chunker.chunk_document(doc, chunk_size=100, overlap=10)
    table_chunks = [c for c in chunks if c.kind == "table"]

    assert len(table_chunks) == 1  # 1 table = 1 chunk (never split)
    t = table_chunks[0]
    assert "Poste | 2022" in t.text and "Chiffre d'affaires | 1000" in t.text
    assert t.page_number == 1 and t.company == "acme" and t.year == 2022
    # the global index covers text AND tables, with no gap
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # order: the page's text chunk(s) precede its table
    assert chunks[-1].kind == "table"


# --- format_chunk_summary: pure summary function -----------------------------------------


def _chunk(text: str, kind: str) -> chunker.Chunk:
    return chunker.Chunk(
        text=text,
        source_path="x.pdf",
        company=None,
        year=None,
        page_number=1,
        chunk_index=0,
        kind=kind,
    )


def test_format_chunk_summary_counts_and_lengths():
    chunks = [_chunk("abcde", "text"), _chunk("fg", "table")]
    # longueurs 5 et 2 → moy = (5+2)//2 = 3, max = 5
    assert chunker.format_chunk_summary(chunks) == (
        "2 chunks (1 texte, 1 tableaux) | long. moy=3 max=5 car."
    )


def test_format_chunk_summary_empty():
    assert chunker.format_chunk_summary([]) == (
        "0 chunks (0 texte, 0 tableaux) | long. moy=0 max=0 car."
    )
