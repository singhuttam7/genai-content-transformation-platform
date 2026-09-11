from app.ingestion.processor import ContentProcessor
from app.ingestion.router import ProcessorRouter
from app.ingestion.schemas import (
    ExtractedContent,
    IngestionRequest,
    InputType,
)


class MockTextProcessor:
    """Minimal processor used only for router testing."""

    supported_types = (InputType.TEXT,)

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        raise NotImplementedError


class MockDocumentProcessor:
    """Minimal processor used only for router testing."""

    supported_types = (
        InputType.PDF,
        InputType.DOCX,
    )

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        raise NotImplementedError


def test_router():
    text_processor = MockTextProcessor()
    document_processor = MockDocumentProcessor()

    router = ProcessorRouter(
        processors=[
            text_processor,
            document_processor,
        ]
    )

    # ---------------------------------------------------------
    # TEXT
    # ---------------------------------------------------------

    selected = router.get_processor(
        InputType.TEXT
    )

    print(
        "TEXT routing:",
        "OK"
        if selected is text_processor
        else "FAILED",
    )

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

    selected = router.get_processor(
        InputType.PDF
    )

    print(
        "PDF routing:",
        "OK"
        if selected is document_processor
        else "FAILED",
    )

    # ---------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------

    selected = router.get_processor(
        InputType.DOCX
    )

    print(
        "DOCX routing:",
        "OK"
        if selected is document_processor
        else "FAILED",
    )

    # ---------------------------------------------------------
    # Unsupported type
    # ---------------------------------------------------------

    print(
        "IMAGE supported:",
        router.supports(InputType.IMAGE),
    )

    print(
        "Supported types:",
        [
            input_type.value
            for input_type in router.supported_types
        ],
    )


if __name__ == "__main__":
    test_router()