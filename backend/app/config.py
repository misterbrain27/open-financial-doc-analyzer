"""Application configuration, driven by the APP_ENV environment variable.

`APP_ENV` ∈ {local, debug, test, prod}. In a container, variables are injected by Docker
Compose; when running on the host, they are read from `.env.<APP_ENV>`.
pydantic-settings precedence: environment variables > .env file > default values.
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.getenv("APP_ENV", "local")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{APP_ENV}"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = APP_ENV
    log_level: str = "INFO"

    # Database (host `db` in container, `localhost` when running on host)
    database_url: str = "postgresql+asyncpg://rag:rag@db:5432/financialrag"

    # Embeddings — Ollama, native on the host
    ollama_base_url: str = "http://host.docker.internal:11434"
    embedding_model: str = "bge-m3"

    # Generation — active provider: "groq" or "ollama"
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Directory for PDFs uploaded via /ingest (relative to the process's WORKDIR;
    # mounted on the host's `./data/raw` in local/debug — see docker-compose.override.yml)
    upload_dir: str = "data/raw"

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


settings = Settings()
