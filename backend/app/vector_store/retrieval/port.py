from __future__ import annotations

from abc import ABC, abstractmethod

from app.vector_store.retrieval.schemas import (
    VectorRetrievalRequest,
    VectorRetrievalResult,
)


class VectorRetrievalPort(ABC):
    """Port for semantic vector retrieval."""

    @abstractmethod
    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        """Search for the most similar vectors."""
        raise NotImplementedError