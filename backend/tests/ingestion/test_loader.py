"""Tests for the PDF loader (Phase 1).

Two families of tests:
- **pure functions** (`clean_text`, `extract_metadata_from_filename`) are tested directly with
  strings / `Path` — no PDF needed;
- `load_pdf` needs a real PDF, **generated on the fly** with `reportlab` in a temporary
  directory (`tmp_path`). No committed binary, no report to download.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, TableStyle
from reportlab.platypus import Table as RLTable

from app.ingestion import loader

# --- Pure functions: no PDF needed -----------------------------------------------------------


def test_clean_text_collapses_spaces_and_tabs():
    assert loader.clean_text("a\t\tb   c") == "a b c"


def test_clean_text_collapses_multiple_newlines():
    # 3 or more consecutive newlines → collapsed to 2
    assert loader.clean_text("l1\n\n\n\nl2") == "l1\n\nl2"


def test_clean_text_empty_string():
    assert loader.clean_text("") == ""


@pytest.mark.parametrize(
    ("filename", "expected_company", "expected_year"),
    [
        ("totalenergies_2023.pdf", "totalenergies", 2023),  # nominal case
        ("document.pdf", "document", None),  # no year in the name
        ("2021.pdf", None, 2021),  # year only → no company name
    ],
)
def test_extract_metadata_from_filename(filename, expected_company, expected_year):
    company, year = loader.extract_metadata_from_filename(Path(filename))
    assert company == expected_company
    assert year == expected_year


# --- load_pdf: PDF generated on the fly ------------------------------------------------------


@pytest.fixture
def pdf_two_pages(tmp_path: Path) -> Path:
    """Generates a 2-page PDF named `acme_2022.pdf` in a temporary directory."""
    chemin = tmp_path / "acme_2022.pdf"
    c = canvas.Canvas(str(chemin))
    c.drawString(72, 800, "Bilan ACME 2022")
    c.showPage()  # end of page 1
    c.drawString(72, 800, "Compte resultat")
    c.showPage()  # end of page 2
    c.save()
    return chemin


def test_load_pdf_reads_pages_and_metadata(pdf_two_pages: Path):
    doc = loader.load_pdf(pdf_two_pages)

    # metadata inferred from the filename
    assert doc.company == "acme"
    assert doc.year == 2022

    # pages: correct count + 1-indexed numbering
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2

    # extracted and cleaned text
    assert "Bilan" in doc.pages[0].text
    assert "Compte" in doc.pages[1].text
    assert "Bilan" in doc.full_text and "Compte" in doc.full_text


def test_load_pdf_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        loader.load_pdf(tmp_path / "n_existe_pas.pdf")


def test_clean_table_collapses_and_strips_whitespace():
    table = [["  Chiffre   d'affaires ", "Actif\nimmobilisé"]]
    assert loader.clean_table(table) == [["Chiffre d'affaires", "Actif immobilisé"]]


def test_clean_table_preserves_none_distinct_from_empty():
    # case B: ["Total", "   ", None] → ["Total", "", None]
    table = [["Total", "   ", None]]
    assert loader.clean_table(table) == [["Total", "", None]]


def test_clean_table_drops_fully_empty_rows():
    # case C: all-None / all-empty rows dropped, partial row kept
    table = [
        [None, None],
        ["", ""],
        ["Total", "   ", None],
        ["", "Valeur"],
    ]
    assert loader.clean_table(table) == [["Total", "", None], ["", "Valeur"]]


# --- format_summary: LoadedDocument built by hand ---------------------------------------------


def test_format_summary_one_line():
    doc = loader.LoadedDocument(
        source_path="/data/raw/novatech_2025.pdf",
        company="novatech",
        year=2025,
        pages=[
            loader.PageContent(page_number=1, text="abcde", tables=[[["a", "b"]]]),
            loader.PageContent(page_number=2, text="fgh", tables=[]),
        ],
    )
    # nb_chars counts `full_text` = "abcde\n\nfgh" -> 10 (the \n\n separator counts).
    assert loader.format_summary(doc) == (
        "novatech_2025.pdf | entreprise=novatech | année=2025 | 2 pages | 1 tableaux | 10 car."
    )


def test_format_summary_handles_missing_metadata():
    # company/year at None -> falls back to "?" ; document with no page -> 0 everywhere.
    doc = loader.LoadedDocument(source_path="x.pdf", company=None, year=None, pages=[])
    assert loader.format_summary(doc) == (
        "x.pdf | entreprise=? | année=? | 0 pages | 0 tableaux | 0 car."
    )


# --- load_pdf: extracting a real table ---------------------------------------------------------


@pytest.fixture
def pdf_with_table(tmp_path: Path) -> Path:
    """Generates `acme_2022.pdf` containing a real table (drawn grid)."""
    chemin = tmp_path / "acme_2022.pdf"
    data = [["Poste", "2022"], ["Chiffre d'affaires", "1000"], ["Résultat net", "200"]]
    t = RLTable(data)
    # pdfplumber detects tables by their lines: without a GRID style, `extract_tables()`
    # finds nothing. So we draw a black grid around each cell.
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    SimpleDocTemplate(str(chemin)).build([t])
    return chemin


def test_load_pdf_extracts_and_cleans_table(pdf_with_table: Path):
    doc = loader.load_pdf(pdf_with_table)
    assert doc.pages[0].tables  # at least one table detected on page 1
    table = doc.pages[0].tables[0]
    assert ["Poste", "2022"] in table
    assert ["Chiffre d'affaires", "1000"] in table
    assert ["Résultat net", "200"] in table
