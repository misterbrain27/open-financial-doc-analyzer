"""Configuration applicative, pilotée par la variable d'environnement APP_ENV.

`APP_ENV` ∈ {local, debug, test, prod}. En conteneur, les variables sont injectées par Docker
Compose ; en exécution sur l'hôte, elles sont lues depuis `.env.<APP_ENV>`.
Précédence pydantic-settings : variables d'environnement > fichier .env > valeurs par défaut.
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

    # Base de données (host `db` en conteneur, `localhost` en exécution hôte)
    database_url: str = "postgresql+asyncpg://rag:rag@db:5432/financialrag"

    # Embeddings — Ollama, en natif sur l'hôte
    ollama_base_url: str = "http://host.docker.internal:11434"
    embedding_model: str = "bge-m3"

    # Génération — fournisseur actif : "mistral" ou "ollama"
    llm_provider: str = "mistral"
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"

    # Dossier de dépôt des PDF uploadés via /ingest (relatif au WORKDIR du process ;
    # monté sur `./data/raw` de l'hôte en local/debug — cf. docker-compose.override.yml)
    upload_dir: str = "data/raw"

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


settings = Settings()


settings = Settings()
