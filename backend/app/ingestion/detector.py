from typing import Protocol

from app.ingestion.schemas import InputType, IngestionRequest


class InputDetector(Protocol):
    """Contract for detecting the type of incoming content."""

    def detect(
        self,
        request: IngestionRequest,
    ) -> InputType:
        """Determine the appropriate input type."""
        ...