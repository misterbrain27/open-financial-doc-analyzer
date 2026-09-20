"""Routes API RAG (Phase 6) : `/ingest` (upload PDF) et `/query` (streaming SSE).

`/ingest` persiste le PDF uploadé sous `settings.upload_dir` (nom de fichier conservé —
`load_pdf` en déduit `company`/`year`, cf. `ingestion/loader.py`) puis délègue à
`ingestion/pipeline.ingest_pdf` (Phase 3).

`/query` streame la réponse du LLM en Server-Sent Events : un event `sources` (JSON, déjà
connu avant la génération car issu de la recherche — Phase 4), puis un event `delta` par
fragment de texte généré, puis `done` — ou `error` si le LLM échoue en cours de streaming
(ex. quota Mistral dépassé), pour ne pas laisser la réponse HTTP se couper sans explication.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.ingestion.pipeline import ingest_pdf
from app.rag.pipeline import answer_query

router = APIRouter()


class DocumentResponse(BaseModel):
    """Réponse de `/ingest` : document créé (Phase 3)."""

    id: int
    source_path: str
    company: str | None
    year: int | None


@router.post(
    "/ingest",
    tags=["ingestion"],
    summary="Ingère un PDF (extraction, chunking, embeddings, pgvector)",
)
async def ingest(
    file: UploadFile, session: AsyncSession = Depends(get_session)
) -> DocumentResponse:
    """Sauvegarde le PDF uploadé puis l'ingère (chunking + embeddings + pgvector)."""
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers .pdf sont acceptés")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename
    destination.write_bytes(await file.read())

    try:
        document = await ingest_pdf(destination, session)
    except Exception as exc:
        # Large volontairement (parsing PDF, chunking, appel embeddings...) : à ce stade on ne
        # distingue pas la cause, on renvoie un 400 plutôt qu'un 500 brut côté client.
        raise HTTPException(400, f"Échec de l'ingestion : {exc}") from exc

    return DocumentResponse(
        id=document.id,
        source_path=document.source_path,
        company=document.company,
        year=document.year,
    )


class QueryRequest(BaseModel):
    """Corps de `/query`."""

    query: str
    k: int = 5
    company: str | None = None
    year: int | None = None


def _sse_event(event: str, data: dict[str, object]) -> str:
    """Formatte un event SSE : `event: <type>\\ndata: <json>\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_answer(request: QueryRequest, session: AsyncSession) -> AsyncIterator[str]:
    try:
        result = await answer_query(
            request.query,
            session,
            k=request.k,
            company=request.company,
            year=request.year,
        )
        yield _sse_event("sources", {"sources": [asdict(source) for source in result.sources]})
        async for delta in result.stream:
            yield _sse_event("delta", {"text": delta})
        yield _sse_event("done", {})
    except Exception as exc:
        # Ex. quota/réseau Mistral en cours de streaming : la réponse HTTP a déjà commencé
        # (status 200 envoyé), on ne peut plus renvoyer un code d'erreur — on notifie via SSE.
        yield _sse_event("error", {"message": str(exc)})


@router.post(
    "/query",
    tags=["query"],
    summary="Question groundée sur les documents ingérés (réponse en streaming SSE)",
)
async def query(
    request: QueryRequest, session: AsyncSession = Depends(get_session)
) -> StreamingResponse:
    return StreamingResponse(_stream_answer(request, session), media_type="text/event-stream")
