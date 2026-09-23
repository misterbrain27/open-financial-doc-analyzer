"""RAG API routes (Phase 6): `/ingest` (PDF upload) and `/query` (SSE streaming).

`/ingest` persists the uploaded PDF under `settings.upload_dir` (filename kept as-is —
`load_pdf` derives `company`/`year` from it, see `ingestion/loader.py`) then delegates to
`ingestion/pipeline.ingest_pdf` (Phase 3).

`/query` streams the LLM's response as Server-Sent Events: a `sources` event (JSON, already
known before generation since it comes from retrieval — Phase 4), then a `delta` event per
generated text fragment, then `done` — or `error` if the LLM fails mid-stream
(e.g. Mistral quota exceeded), so the HTTP response doesn't cut off without explanation.
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
    """Response of `/ingest`: created document (Phase 3)."""

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
    """Saves the uploaded PDF then ingests it (chunking + embeddings + pgvector)."""
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers .pdf sont acceptés")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename
    destination.write_bytes(await file.read())

    try:
        document = await ingest_pdf(destination, session)
    except Exception as exc:
        # Deliberately broad (PDF parsing, chunking, embeddings call...): at this stage we don't
        # distinguish the cause, we return a 400 rather than a raw 500 to the client.
        raise HTTPException(400, f"Échec de l'ingestion : {exc}") from exc

    return DocumentResponse(
        id=document.id,
        source_path=document.source_path,
        company=document.company,
        year=document.year,
    )


class QueryRequest(BaseModel):
    """Body of `/query`."""

    query: str
    k: int = 5
    company: str | None = None
    year: int | None = None


def _sse_event(event: str, data: dict[str, object]) -> str:
    """Formats an SSE event: `event: <type>\\ndata: <json>\\n\\n`."""
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
        # E.g. Mistral quota/network issue mid-streaming: the HTTP response has already started
        # (status 200 sent), we can no longer return an error code — we notify via SSE.
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
