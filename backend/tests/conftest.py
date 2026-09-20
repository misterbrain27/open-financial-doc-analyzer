"""Shared fixtures: real test database (Postgres + pgvector, cf. docker-compose.test.yml).

The schema is (re)created before each test (`init_db` is idempotent — `CREATE EXTENSION IF NOT
EXISTS` + `create_all` with `checkfirst`). Each test receives a session bound to a connection
whose transaction is rolled back at the end (`db_session`), so that no data persists from one
test to another.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import engine, init_db


@pytest.fixture(scope="session")
async def _setup_db() -> None:
    await init_db()


@pytest.fixture
async def db_session(_setup_db: None) -> AsyncIterator[AsyncSession]:
    async with engine.connect() as conn:
        await conn.begin()
        session_factory = async_sessionmaker(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        async with session_factory() as session:
            yield session
        await conn.rollback()
