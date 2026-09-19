"""Schéma de la base de données : tables `documents` et `chunks`.

`Chunk.embedding` est un vecteur pgvector de dimension `EMBEDDING_DIM` (cf.
`app/ingestion/embedder.py`). L'index HNSW (`vector_cosine_ops`) accélère la recherche par
similarité cosinus (Phase 4) — sans lui, chaque requête comparerait le vecteur de la question à
tous les vecteurs stockés (scan complet).
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.ingestion.embedder import EMBEDDING_DIM


class Document(Base):
    """Un document source (rapport annuel / document AMF) chargé en Phase 1."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_path: Mapped[str]
    company: Mapped[str | None]
    year: Mapped[int | None]


class Chunk(Base):
    """Un fragment de document (texte ou tableau, Phase 2) avec son embedding (Phase 3)."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    text: Mapped[str]
    page_number: Mapped[int]
    chunk_index: Mapped[int]
    kind: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
