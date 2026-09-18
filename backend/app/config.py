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


settings = Settings()
