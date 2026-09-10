from __future__ import annotations

from typing import Protocol

from app.ingestion.schemas import IngestionRequest


class InputContentResolver(Protocol):
    """Resolve an ingestion request into binary content."""

    async def resolve(self, request: IngestionRequest) -> bytes:
        """Return the input content as bytes."""
        ...