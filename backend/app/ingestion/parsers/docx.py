from __future__ import annotations

from io import BytesIO

from docx import Document
from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph

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


class DOCXProcessor(ContentProcessor):
    """
    Extract structured textual content from DOCX documents.

    Responsibilities:
    - Resolve DOCX bytes through InputContentResolver.
    - Extract paragraphs, headings, lists, and tables.
    - Preserve document element ordering.
    - Preserve basic structural metadata.
    - Return ExtractedContent for downstream normalization.

    AI enrichment and content transformation are intentionally
    outside this processor.
    """

    supported_types = frozenset({InputType.DOCX})

    def __init__(self, resolver: InputContentResolver) -> None:
        self.resolver = resolver

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        """Process a DOCX ingestion request."""

        if request.input_type != InputType.DOCX:
            raise ValueError(
                "DOCXProcessor can only process DOCX inputs."
            )

        docx_bytes = await self.resolver.resolve(request)

        if not docx_bytes:
            raise ValueError(
                "DOCX content cannot be empty."
            )

        try:
            document = Document(BytesIO(docx_bytes))
        except Exception as exc:
            raise ValueError(
                "Unable to read DOCX document."
            ) from exc

        blocks = self._extract_blocks(document)

        if not blocks:
            raise ValueError(
                "DOCX document contains no extractable "
                "textual content."
            )

        full_text = "\n\n".join(
            block.content
            for block in blocks
            if block.content.strip()
        )

        if not full_text.strip():
            raise ValueError(
                "DOCX document contains no extractable "
                "textual content."
            )

        source = SourceReference(
            source_id=request.source_id,
            source_type=InputType.DOCX,
            title=request.title,
            filename=request.filename,
            mime_type=request.mime_type,
            storage_uri=request.storage_uri,
        )

        return ExtractedContent(
            source=source,
            title=request.title,
            text=full_text,
            blocks=blocks,
            language=None,
            metadata={
                **request.metadata,
                "processor": "docx",
                "block_count": len(blocks),
            },
        )

    @classmethod
    def _extract_blocks(
        cls,
        document: DocumentType,
    ) -> list[ContentBlock]:
        """
        Extract paragraphs and tables while preserving
        their document order.
        """

        blocks: list[ContentBlock] = []

        for element in document.element.body.iterchildren():

            if element.tag.endswith("}p"):
                paragraph = Paragraph(
                    element,
                    document,
                )

                block = cls._paragraph_to_block(
                    paragraph=paragraph,
                    order=len(blocks),
                )

                if block is not None:
                    blocks.append(block)

            elif element.tag.endswith("}tbl"):
                table = Table(
                    element,
                    document,
                )

                block = cls._table_to_block(
                    table=table,
                    order=len(blocks),
                )

                if block is not None:
                    blocks.append(block)

        return blocks

    @staticmethod
    def _paragraph_to_block(
        *,
        paragraph: Paragraph,
        order: int,
    ) -> ContentBlock | None:
        """Convert a DOCX paragraph into a content block."""

        text = paragraph.text.strip()

        if not text:
            return None

        style_name = (
            paragraph.style.name.strip()
            if paragraph.style is not None
            else ""
        )

        style_lower = style_name.lower()

        if style_lower.startswith("heading"):
            block_type = ContentBlockType.HEADING

            heading_level = (
                DOCXProcessor._extract_heading_level(
                    style_name
                )
            )

            metadata = {
                "processor": "docx",
                "style": style_name,
                "heading_level": heading_level,
            }

        elif DOCXProcessor._is_list_paragraph(paragraph):
            block_type = ContentBlockType.LIST

            metadata = {
                "processor": "docx",
                "style": style_name,
                "list": True,
            }

        else:
            block_type = ContentBlockType.PARAGRAPH

            metadata = {
                "processor": "docx",
                "style": style_name,
            }

        return ContentBlock(
            block_type=block_type,
            content=text,
            order=order,
            metadata=metadata,
        )

    @staticmethod
    def _table_to_block(
        *,
        table: Table,
        order: int,
    ) -> ContentBlock | None:
        """Convert a DOCX table into a content block."""

        rows: list[list[str]] = []

        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            rows.append(cells)

        if not rows:
            return None

        non_empty_rows = [
            row
            for row in rows
            if any(
                cell.strip()
                for cell in row
            )
        ]

        if not non_empty_rows:
            return None

        table_text = "\n".join(
            " | ".join(row)
            for row in non_empty_rows
        )

        return ContentBlock(
            block_type=ContentBlockType.TABLE,
            content=table_text,
            order=order,
            metadata={
                "processor": "docx",
                "table_index": order,
                "rows": len(non_empty_rows),
                "columns": max(
                    len(row)
                    for row in non_empty_rows
                ),
            },
        )

    @staticmethod
    def _extract_heading_level(
        style_name: str,
    ) -> int | None:
        """
        Extract heading level from styles such as:

        Heading 1
        Heading 2
        Heading 3
        """

        parts = style_name.split()

        if len(parts) != 2:
            return None

        if parts[0].lower() != "heading":
            return None

        try:
            level = int(parts[1])
        except ValueError:
            return None

        if 1 <= level <= 9:
            return level

        return None

    @staticmethod
    def _is_list_paragraph(
        paragraph: Paragraph,
    ) -> bool:
        """
        Detect common Word list paragraphs.

        Lists may be represented either through the
        paragraph's numbering properties or through a
        built-in list style.
        """

        paragraph_properties = paragraph._p.pPr

        if paragraph_properties is not None:
            if paragraph_properties.numPr is not None:
                return True

        style_name = (
            paragraph.style.name.strip()
            if paragraph.style is not None
            else ""
        )

        style_lower = style_name.lower()

        list_style_prefixes = (
            "list",
            "bullet",
            "number",
        )

        return style_lower.startswith(
            list_style_prefixes
        )