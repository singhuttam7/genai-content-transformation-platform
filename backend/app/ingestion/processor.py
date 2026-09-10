from typing import Protocol

from app.ingestion.schemas import (
    ExtractedContent,
    IngestionRequest,
)


class ContentProcessor(Protocol):
    """Contract implemented by every ingestion processor."""

    supported_types: frozenset

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        """Extract structured content from the input."""
        ...