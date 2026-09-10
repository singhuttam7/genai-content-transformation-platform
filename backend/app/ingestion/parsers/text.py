from __future__ import annotations

from app.ingestion.processor import ContentProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)


class TextProcessor:
    """Processor for plain text and free-form prompts."""

    supported_types = (
        InputType.TEXT,
        InputType.PROMPT,
    )

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        """Extract structured content from text input."""

        if request.content is None:
            raise ValueError(
                "Text content is required."
            )

        if isinstance(request.content, bytes):
            text = request.content.decode(
                "utf-8",
                errors="replace",
            )
        else:
            text = str(request.content)

        text = text.strip()

        if not text:
            raise ValueError(
                "Text content cannot be empty."
            )

        source = SourceReference(
            source_id=request.source_id,
            source_type=request.input_type,
            title=request.title,
            filename=request.filename,
            mime_type=request.mime_type,
        )

        block = ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content=text,
            order=0,
        )

        return ExtractedContent(
            source=source,
            title=request.title,
            text=text,
            blocks=[block],
            language=None,
            metadata=request.metadata,
        )