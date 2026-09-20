"""Database schema: `documents` and `chunks` tables.

`Chunk.embedding` is a pgvector vector of dimension `EMBEDDING_DIM` (see
`app/ingestion/embedder.py`). The HNSW index (`vector_cosine_ops`) speeds up cosine
similarity search (Phase 4) — without it, each query would compare the question's vector
against every stored vector (full scan).

`Chunk.tsv` is a generated column (full-text `french`) maintained by Postgres and indexed
with GIN: it carries the LEXICAL signal of hybrid search (Phase 9), complementary to the
vector one.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.ingestion.embedder import EMBEDDING_DIM


class Document(Base):
    """A source document (annual report / AMF filing) loaded in Phase 1."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_path: Mapped[str]
    company: Mapped[str | None]
    year: Mapped[int | None]


class Chunk(Base):
    """A document fragment (text or table, Phase 2) with its embedding (Phase 3)."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    text: Mapped[str]
    page_number: Mapped[int]
    chunk_index: Mapped[int]
    kind: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    # Full-text representation of `text`, maintained by Postgres (STORED generated column). Carries
    # the LEXICAL signal of hybrid search (Phase 9); never read on the Python side (hence the
    # nominal `Mapped[str]` annotation), it only exists for SQL (`ts_rank_cd`, `@@`), sped up by
    # the GIN index below.
    # ⚠️ `to_tsvector('french', text)` in the 2-ARGUMENT form: a generated column requires an
    # immutable expression; the 1-argument form depends on `default_text_search_config` (a
    # session setting) and would be rejected by Postgres.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('french', text)", persisted=True),
    )

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index on the full-text column: speeds up the lexical signal, just as the HNSW above
        # speeds up the vector one.
        Index("ix_chunks_tsv_gin", "tsv", postgresql_using="gin"),
    )
