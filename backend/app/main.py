"""FastAPI entry point.

Minimal application (`/health` probe) enriched in Phase 6 with the RAG routes (`/ingest`,
`/query`) — see `app/api/routes.py`.

Documentation: Swagger UI is served at /docs with a dark theme. FastAPI generates
the whole OpenAPI spec automatically; we only disable the default /docs to
re-inject our dark stylesheet (see `swagger_ui_dark`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.routes import router as api_router
from app.config import settings
from app.db import init_db

DESCRIPTION = """
API du **financial-doc-analyzer** — système RAG d'analyse de documents financiers
français (rapports annuels / AMF).

Architecture hybride : embeddings locaux (Ollama `bge-m3`) + génération via Mistral AI,
stockage vectoriel PostgreSQL + pgvector.

`/ingest` charge un PDF en base ; `/query` interroge les documents ingérés (réponse
streamée en SSE, avec citation des sources).
"""


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Enables pgvector and creates the tables on startup (`create_all`, idempotent).

    Without this hook, a fresh database (new volume, `make clean`) has no schema:
    `/ingest`/`/query` would fail on non-existent tables until
    `scripts/ingest_cli.py` has been run manually.
    """
    await init_db()
    yield


# `docs_url=None`: we take over /docs to inject the dark CSS
# (see the `swagger_ui_dark` route below). /redoc keeps the default theme.
app = FastAPI(
    title="Financial Doc Analyzer API",
    description=DESCRIPTION,
    version="0.1.0",
    docs_url=None,
    lifespan=lifespan,
    contact={"name": "David", "url": "https://github.com/misterbrain27/financial-doc-analyzer"},
    openapi_tags=[
        {"name": "health", "description": "Sondes de disponibilité et environnement actif."},
        {"name": "ingestion", "description": "Chargement de documents PDF en base."},
        {"name": "query", "description": "Interrogation RAG des documents ingérés."},
    ],
)

app.include_router(api_router)

# Static assets (Swagger theme CSS). `Path(__file__).parent` resolves correctly
# both on host execution and inside the container (directory /app/app/static).
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/docs", include_in_schema=False)
async def swagger_ui_dark() -> HTMLResponse:
    """Swagger UI with a dark theme (CSS served from /static)."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=f"{app.title} — docs",
        swagger_css_url="/static/swagger-dark.css",
    )


class HealthResponse(BaseModel):
    """Response of the availability probe."""

    status: str
    env: str


@app.get("/health", tags=["health"], summary="Sonde de disponibilité")
async def health() -> HealthResponse:
    """Checks that the application is responding and exposes the active environment."""
    return HealthResponse(status="ok", env=settings.app_env)
