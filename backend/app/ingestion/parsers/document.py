from __future__ import annotations

from app.ingestion.content_resolver import InputContentResolver
from app.ingestion.processor import ContentProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)


class TextDocumentProcessor(ContentProcessor):
    """
    Processor for TXT and Markdown documents.

    Responsibilities:
    - Resolve document bytes through InputContentResolver.
    - Decode UTF-8 content.
    - Preserve source text faithfully.
    - Create deterministic content blocks.
    - Return ExtractedContent.
    """

    supported_types = (
        InputType.TXT,
        InputType.MARKDOWN,
    )

    def __init__(
        self,
        resolver: InputContentResolver,
    ) -> None:
        self.resolver = resolver

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        content = await self.resolver.resolve(request)

        text = self._decode_utf8(content)

        if not text.strip():
            raise ValueError(
                "Document content cannot be empty."
            )

        blocks = self._build_blocks(
            text=text,
            input_type=request.input_type,
        )

        source = SourceReference(
            source_id=request.source_id,
            source_type=request.input_type,
            title=request.title,
            filename=request.filename,
            mime_type=request.mime_type,
            storage_uri=request.storage_uri,
        )

        return ExtractedContent(
            source=source,
            title=request.title,
            text=text,
            blocks=blocks,
            language=None,
            metadata=request.metadata,
        )

    @staticmethod
    def _decode_utf8(content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Document must contain valid UTF-8 text."
            ) from exc

    @staticmethod
    def _build_blocks(
        *,
        text: str,
        input_type: InputType,
    ) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []

        lines = text.splitlines()

        current_lines: list[str] = []
        order = 0

        def flush_paragraph() -> None:
            nonlocal order

            if not current_lines:
                return

            paragraph = "\n".join(
                current_lines
            ).strip()

            if paragraph:
                blocks.append(
                    ContentBlock(
                        block_type=ContentBlockType.PARAGRAPH,
                        content=paragraph,
                        order=order,
                    )
                )
                order += 1

            current_lines.clear()

        for line in lines:
            stripped = line.strip()

            # Blank line separates paragraphs.
            if not stripped:
                flush_paragraph()
                continue

            # Markdown heading.
            if (
                input_type == InputType.MARKDOWN
                and stripped.startswith("#")
            ):
                flush_paragraph()

                heading_text = stripped.lstrip("#").strip()

                if heading_text:
                    blocks.append(
                        ContentBlock(
                            block_type=ContentBlockType.HEADING,
                            content=heading_text,
                            order=order,
                            metadata={
                                "markdown": True,
                            },
                        )
                    )
                    order += 1

                continue

            current_lines.append(
                line.rstrip()
            )

        flush_paragraph()

        return blocks