"""Orchestration bout-en-bout (Phase 3) : PDF → chunks → embeddings → BDD pgvector.

Enchaîne les trois étapes déjà écrites (`load_pdf`, `chunk_document`, `embed_texts`) et
persiste le résultat via SQLAlchemy (`Document` + `Chunk`, cf. `app/models.py`).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.ingestion.loader import load_pdf
from app.models import Chunk, Document


async def ingest_pdf(
    path: str | Path, session: AsyncSession, *, chunk_size: int = 1000, overlap: int = 150
) -> Document:
    """Charge, découpe, embarque et persiste un PDF. Renvoie le `Document` créé."""
    doc = load_pdf(path)
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
    embeddings = await embed_texts([chunk.text for chunk in chunks])

    document = Document(source_path=doc.source_path, company=doc.company, year=doc.year)
    session.add(document)
    await session.flush()  # attribue document.id (auto-incrément) sans committer

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        session.add(
            Chunk(
                document_id=document.id,
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                kind=chunk.kind,
                embedding=embedding,
            )
        )

    await session.commit()
    return document
