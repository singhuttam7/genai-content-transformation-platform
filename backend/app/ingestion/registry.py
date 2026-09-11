from __future__ import annotations

from app.ingestion.parsers.docx import DOCXProcessor
from app.ingestion.parsers.document import TextDocumentProcessor
from app.ingestion.parsers.image import ImageDocumentProcessor
from app.ingestion.parsers.pdf import PDFProcessor
from app.ingestion.parsers.text import TextProcessor
from app.ingestion.router import ProcessorRouter
from app.ingestion.storage_resolver import StorageBackedContentResolver
from app.storage.dependencies import get_storage_service


def create_processor_router() -> ProcessorRouter:
    """Create the application processor router."""

    storage = get_storage_service()

    resolver = StorageBackedContentResolver(
        storage=storage,
    )

    processors = [
        TextProcessor(),
        TextDocumentProcessor(
            resolver=resolver,
        ),
        PDFProcessor(
            resolver=resolver,
        ),
        DOCXProcessor(
            resolver=resolver,
        ),
        ImageDocumentProcessor(),
    ]

    return ProcessorRouter(
        processors=processors,
    )