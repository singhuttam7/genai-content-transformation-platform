from __future__ import annotations

import pytest
import pytest_asyncio

from app.database.connection import engine


@pytest_asyncio.fixture(autouse=True)
async def cleanup_database_engine() -> None:
    """Dispose pooled database connections after every async test."""

    yield

    await engine.dispose()