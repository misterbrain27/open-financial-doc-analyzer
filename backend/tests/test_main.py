"""Tests for the FastAPI app: /health probe, dark Swagger docs, and OpenAPI metadata.

Uses `TestClient` (httpx under the hood, already a dependency) which mounts the app without
starting a server.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # matches the HealthResponse response_model
    assert body["status"] == "ok"
    assert "env" in body


def test_docs_serves_swagger_with_dark_css():
    resp = client.get("/docs")
    assert resp.status_code == 200
    # the Swagger page must reference our dark stylesheet
    assert "/static/swagger-dark.css" in resp.text


def test_dark_css_is_served_and_imports_base():
    resp = client.get("/static/swagger-dark.css")
    assert resp.status_code == 200
    # the key mechanism: the CSS first imports the base swagger-ui stylesheet
    assert "@import" in resp.text
    assert "swagger-ui-dist" in resp.text


def test_openapi_exposes_enriched_metadata():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    info = resp.json()["info"]
    assert info["title"] == "Financial Doc Analyzer API"
    assert info["description"]  # non-empty description
