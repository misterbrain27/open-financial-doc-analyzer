"""Database layer: async engine, session factory, declarative base.

`init_db()` enables the pgvector extension and creates the tables (no Alembic migrations at
this stage of the project — see `models.py` for the schema).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Enables pgvector and creates the tables declared on `Base` if they don't exist."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency (`Depends(get_session)`): one session per request.

    Commits if the route executed without error, rolls back otherwise. Tests
    replace this dependency (`app.dependency_overrides`) with the transactional
    `db_session` fixture, so test data is never persisted.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
