from __future__ import annotations

from app.ingestion.parsers.text import TextProcessor
from app.ingestion.router import ProcessorRouter


def create_processor_router() -> ProcessorRouter:
    """Create the application ingestion processor router."""

    return ProcessorRouter(
        processors=[
            TextProcessor(),
        ]
    )