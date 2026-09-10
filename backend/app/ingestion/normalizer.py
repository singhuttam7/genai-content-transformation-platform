from typing import Protocol

from app.ingestion.schemas import (
    CanonicalContent,
    ExtractedContent,
)


class ContentNormalizer(Protocol):
    """Contract for converting extracted content into canonical content."""

    async def normalize(
        self,
        content: ExtractedContent,
    ) -> CanonicalContent:
        """Normalize extracted content into the platform representation."""
        ...