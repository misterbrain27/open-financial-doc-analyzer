"""Point d'entrée FastAPI.

Application minimale (sonde `/health`) enrichie en Phase 6 des routes RAG (`/ingest`,
`/query`) — voir `app/api/routes.py`.

Documentation : Swagger UI est servi à /docs avec un thème sombre. FastAPI génère
tout l'OpenAPI automatiquement ; on désactive seulement le /docs par défaut pour
réinjecter notre feuille de style sombre (voir `swagger_ui_dark`).
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
    """Active pgvector et crée les tables au démarrage (`create_all`, idempotent).

    Sans ce hook, une base fraîche (nouveau volume, `make clean`) n'a aucun schéma :
    `/ingest`/`/query` échoueraient sur des tables inexistantes tant que
    `scripts/ingest_cli.py` n'a pas été lancé manuellement.
    """
    await init_db()
    yield


# `docs_url=None` : on reprend la main sur /docs pour injecter la CSS sombre
# (voir la route `swagger_ui_dark` plus bas). /redoc reste au thème par défaut.
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

# Ressources statiques (CSS du thème Swagger). `Path(__file__).parent` résout aussi bien
# en exécution hôte que dans le conteneur (répertoire /app/app/static).
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/docs", include_in_schema=False)
async def swagger_ui_dark() -> HTMLResponse:
    """Swagger UI avec thème sombre (CSS servie depuis /static)."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=f"{app.title} — docs",
        swagger_css_url="/static/swagger-dark.css",
    )


class HealthResponse(BaseModel):
    """Réponse de la sonde de disponibilité."""

    status: str
    env: str


@app.get("/health", tags=["health"], summary="Sonde de disponibilité")
async def health() -> HealthResponse:
    """Vérifie que l'application répond et expose l'environnement actif."""
    return HealthResponse(status="ok", env=settings.app_env)
