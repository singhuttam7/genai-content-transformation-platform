"""PostgreSQL vector retrieval adapters."""

from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)

__all__ = [
    "PostgresVectorRetrieval",
]