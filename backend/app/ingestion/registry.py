from __future__ import annotations

from app.ingestion.parsers.audio import AudioDocumentProcessor
from app.ingestion.parsers.docx import DOCXProcessor
from app.ingestion.parsers.document import TextDocumentProcessor
from app.ingestion.parsers.image import ImageDocumentProcessor
from app.ingestion.parsers.pdf import PDFProcessor
from app.ingestion.parsers.text import TextProcessor

from app.ingestion.router import ProcessorRouter

from app.ingestion.storage_resolver import (
    StorageBackedContentResolver,
)

from app.storage.dependencies import (
    get_storage_service,
)

from app.ingestion.video.processor import (
    VideoDocumentProcessor,
)


def create_processor_router() -> ProcessorRouter:
    """
    Create the application processor router.

    Every supported input type is mapped to a dedicated
    ContentProcessor implementation.

    Current processors:

        TEXT       -> TextProcessor
        TXT        -> TextDocumentProcessor
        MARKDOWN   -> TextDocumentProcessor
        PDF        -> PDFProcessor
        DOCX       -> DOCXProcessor
        IMAGE      -> ImageDocumentProcessor
        AUDIO      -> AudioDocumentProcessor
        VIDEO      -> VideoDocumentProcessor
    """

    # ---------------------------------------------------------
    # Shared storage-backed content resolver
    # ---------------------------------------------------------

    storage = get_storage_service()

    resolver = StorageBackedContentResolver(
        storage=storage,
    )

    # ---------------------------------------------------------
    # Processor registration
    # ---------------------------------------------------------

    processors = [
        # -----------------------------------------------------
        # Text
        # -----------------------------------------------------

        TextProcessor(),

        # -----------------------------------------------------
        # Text documents
        # -----------------------------------------------------

        TextDocumentProcessor(
            resolver=resolver,
        ),

        # -----------------------------------------------------
        # PDF
        # -----------------------------------------------------

        PDFProcessor(
            resolver=resolver,
        ),

        # -----------------------------------------------------
        # DOCX
        # -----------------------------------------------------

        DOCXProcessor(
            resolver=resolver,
        ),

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        ImageDocumentProcessor(),

        # -----------------------------------------------------
        # Audio
        # -----------------------------------------------------

        AudioDocumentProcessor(),

        # -----------------------------------------------------
        # Video
        # -----------------------------------------------------

        VideoDocumentProcessor(),
    ]

    # ---------------------------------------------------------
    # Build router
    # ---------------------------------------------------------

    return ProcessorRouter(
        processors=processors,
    )