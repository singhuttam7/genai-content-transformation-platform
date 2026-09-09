from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import settings


def create_database_engine() -> AsyncEngine:
    """Create the asynchronous SQLAlchemy database engine."""

    return create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )


engine = create_database_engine()