from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

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


class PDFProcessor(ContentProcessor):
    """
    Extract textual content from PDF documents.

    Responsibilities:
    - Resolve PDF bytes through InputContentResolver.
    - Extract text page-by-page using pypdf.
    - Preserve page-level provenance.
    - Return ExtractedContent for downstream normalization.

    OCR is intentionally outside this processor.
    Image-only/scanned PDFs will be handled by the OCR subsystem.
    """

    supported_types = frozenset({InputType.PDF})

    def __init__(
        self,
        resolver: InputContentResolver,
    ) -> None:
        self.resolver = resolver

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        if request.input_type != InputType.PDF:
            raise ValueError(
                "PDFProcessor can only process PDF inputs."
            )

        pdf_bytes = await self.resolver.resolve(request)

        if not pdf_bytes:
            raise ValueError(
                "PDF content cannot be empty."
            )

        try:
            reader = PdfReader(BytesIO(pdf_bytes))
        except PdfReadError as exc:
            raise ValueError(
                "Unable to read PDF document."
            ) from exc
        except Exception as exc:
            raise ValueError(
                "Failed to parse PDF document."
            ) from exc

        if reader.is_encrypted:
            raise ValueError(
                "Encrypted PDF documents are not supported."
            )

        blocks: list[ContentBlock] = []
        page_texts: list[str] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise ValueError(
                    f"Failed to extract text from PDF page "
                    f"{page_number}."
                ) from exc

            text = text.strip()

            if not text:
                continue

            page_texts.append(text)

            blocks.append(
                ContentBlock(
                    block_type=ContentBlockType.PARAGRAPH,
                    content=text,
                    order=len(blocks),
                    page_number=page_number,
                    metadata={
                        "processor": "pdf",
                    },
                )
            )

        if not blocks:
            raise ValueError(
                "PDF contains no extractable text. "
                "The document may be scanned or image-only."
            )

        full_text = "\n\n".join(page_texts)

        source = SourceReference(
            source_id=request.source_id,
            source_type=InputType.PDF,
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
                "processor": "pdf",
                "page_count": len(reader.pages),
                "extracted_page_count": len(blocks),
            },
        )