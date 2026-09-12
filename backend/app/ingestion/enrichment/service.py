from __future__ import annotations

from copy import deepcopy

from app.ingestion.content_resolver import InputContentResolver

from app.ingestion.ocr.provider import OCRProvider
from app.ingestion.ocr.schemas import (
    OCRRequest,
    OCRResult,
    OCRStatus,
)

from app.ingestion.speech.provider import ASRProvider
from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRResult,
    ASRStatus,
)

from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
)


class ContentEnrichmentService:
    """
    Application-level content enrichment service.

    Responsibilities:
    - Determine which enrichment capabilities apply.
    - Resolve required source bytes through InputContentResolver.
    - Invoke provider abstractions.
    - Preserve original extracted content.
    - Add OCR-derived textual blocks.
    - Add ASR-derived transcript blocks.
    - Preserve enrichment metadata.
    - Treat OCR and ASR failures as non-fatal.
    """

    def __init__(
        self,
        *,
        content_resolver: InputContentResolver,
        ocr_provider: OCRProvider | None = None,
        asr_provider: ASRProvider | None = None,
    ) -> None:
        self.content_resolver = content_resolver
        self.ocr_provider = ocr_provider
        self.asr_provider = asr_provider

    # =========================================================
    # Main enrichment entry point
    # =========================================================

    async def enrich(
        self,
        *,
        request: IngestionRequest,
        content: ExtractedContent,
    ) -> ExtractedContent:
        """
        Enrich extracted content using configured capabilities.

        IMAGE -> OCR
        AUDIO -> ASR
        Other input types -> unchanged
        """

        enriched = content.model_copy(deep=True)

        if request.input_type == InputType.IMAGE:
            return await self._enrich_image(
                request=request,
                content=enriched,
            )

        if request.input_type == InputType.AUDIO:
            return await self._enrich_audio(
                request=request,
                content=enriched,
            )

        return enriched

    # =========================================================
    # IMAGE -> OCR
    # =========================================================

    async def _enrich_image(
        self,
        *,
        request: IngestionRequest,
        content: ExtractedContent,
    ) -> ExtractedContent:
        """
        Perform OCR enrichment for image content.
        """

        if self.ocr_provider is None:
            return content

        try:
            image_bytes = await self.content_resolver.resolve(
                request
            )

        except Exception as exc:
            return self._record_ocr_resolution_failure(
                content,
                exc,
            )

        ocr_result = await self.ocr_provider.recognize(
            self._build_ocr_request(
                image_bytes=image_bytes,
                request=request,
            )
        )

        return self._apply_ocr_result(
            content=content,
            result=ocr_result,
        )

    @staticmethod
    def _build_ocr_request(
        *,
        image_bytes: bytes,
        request: IngestionRequest,
    ) -> OCRRequest:
        """
        Build the provider-independent OCR request.
        """

        language = request.metadata.get(
            "ocr_language"
        )

        if language is not None:
            language = str(language)

        return OCRRequest(
            image=image_bytes,
            language=language,
            metadata={
                "source_id": (
                    str(request.source_id)
                    if request.source_id is not None
                    else None
                ),
                "filename": request.filename,
                "mime_type": request.mime_type,
            },
        )

    @staticmethod
    def _apply_ocr_result(
        *,
        content: ExtractedContent,
        result: OCRResult,
    ) -> ExtractedContent:
        """
        Apply OCR result to extracted content.

        OCR blocks are represented as PARAGRAPH blocks while
        preserving the original IMAGE block.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["ocr"] = {
            "status": result.status.value,
            "provider": result.provider,
            "language": result.language,
            "confidence": result.confidence,
            "block_count": len(result.blocks),
            **result.metadata,
        }

        # -----------------------------------------------------
        # Non-successful OCR is non-fatal.
        # -----------------------------------------------------

        if result.status != OCRStatus.COMPLETED:
            return content.model_copy(
                update={
                    "metadata": metadata,
                }
            )

        existing_blocks = list(
            content.blocks
        )

        next_order = (
            max(
                (
                    block.order
                    for block in existing_blocks
                ),
                default=-1,
            )
            + 1
        )

        ocr_blocks: list[ContentBlock] = []

        for offset, ocr_block in enumerate(
            result.blocks
        ):
            bounding_box = None

            if ocr_block.bounding_box is not None:
                bounding_box = {
                    "x": ocr_block.bounding_box.x,
                    "y": ocr_block.bounding_box.y,
                    "width": ocr_block.bounding_box.width,
                    "height": ocr_block.bounding_box.height,
                }

            ocr_blocks.append(
                ContentBlock(
                    block_type=ContentBlockType.PARAGRAPH,
                    content=ocr_block.text,
                    order=next_order + offset,
                    metadata={
                        "source": "ocr",
                        "provider": result.provider,
                        "language": result.language,
                        "confidence": ocr_block.confidence,
                        "bounding_box": bounding_box,
                        **ocr_block.metadata,
                    },
                )
            )

        combined_text_parts: list[str] = []

        if content.text.strip():
            combined_text_parts.append(
                content.text.strip()
            )

        if result.text.strip():
            combined_text_parts.append(
                result.text.strip()
            )

        combined_text = "\n\n".join(
            combined_text_parts
        )

        return content.model_copy(
            update={
                "text": combined_text,
                "blocks": (
                    existing_blocks
                    + ocr_blocks
                ),
                "language": (
                    content.language
                    or result.language
                ),
                "metadata": metadata,
            }
        )

    @staticmethod
    def _record_ocr_resolution_failure(
        content: ExtractedContent,
        error: Exception,
    ) -> ExtractedContent:
        """
        Record OCR source-resolution failure without
        failing the complete ingestion pipeline.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["ocr"] = {
            "status": "not_available",
            "reason": (
                "Image content could not be resolved "
                "for OCR enrichment."
            ),
            "error_type": type(error).__name__,
            "error": str(error),
        }

        return content.model_copy(
            update={
                "metadata": metadata,
            }
        )

    # =========================================================
    # AUDIO -> ASR
    # =========================================================

    async def _enrich_audio(
        self,
        *,
        request: IngestionRequest,
        content: ExtractedContent,
    ) -> ExtractedContent:
        """
        Perform ASR enrichment for audio content.
        """

        if self.asr_provider is None:
            return content

        try:
            audio_bytes = await self.content_resolver.resolve(
                request
            )

        except Exception as exc:
            return self._record_asr_resolution_failure(
                content,
                exc,
            )

        asr_request = self._build_asr_request(
            audio_bytes=audio_bytes,
            request=request,
        )

        asr_result = await self.asr_provider.transcribe(
            asr_request
        )

        return self._apply_asr_result(
            content=content,
            result=asr_result,
        )

    @staticmethod
    def _build_asr_request(
        *,
        audio_bytes: bytes,
        request: IngestionRequest,
    ) -> ASRRequest:
        """
        Build the provider-independent ASR request.
        """

        language = request.metadata.get(
            "asr_language"
        )

        if language is not None:
            language = str(language)

        return ASRRequest(
            audio=audio_bytes,
            language=language,
            metadata={
                "source_id": (
                    str(request.source_id)
                    if request.source_id is not None
                    else None
                ),
                "filename": request.filename,
                "mime_type": request.mime_type,
            },
        )

    @staticmethod
    def _apply_asr_result(
        *,
        content: ExtractedContent,
        result: ASRResult,
    ) -> ExtractedContent:
        """
        Apply ASR result to extracted content.

        Each ASR segment becomes a TRANSCRIPT block
        containing timing and provider metadata.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["asr"] = {
            "status": result.status.value,
            "provider": result.provider,
            "language": result.language,
            "confidence": result.confidence,
            "segment_count": len(result.segments),
            **result.metadata,
        }

        # -----------------------------------------------------
        # Non-successful ASR is non-fatal.
        # -----------------------------------------------------

        if result.status != ASRStatus.COMPLETED:
            return content.model_copy(
                update={
                    "metadata": metadata,
                }
            )

        existing_blocks = list(
            content.blocks
        )

        next_order = (
            max(
                (
                    block.order
                    for block in existing_blocks
                ),
                default=-1,
            )
            + 1
        )

        transcript_blocks: list[ContentBlock] = []

        for offset, segment in enumerate(
            result.segments
        ):
            transcript_blocks.append(
                ContentBlock(
                    block_type=ContentBlockType.TRANSCRIPT,
                    content=segment.text,
                    order=next_order + offset,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    metadata={
                        "source": "asr",
                        "provider": result.provider,
                        "language": result.language,
                        "confidence": segment.confidence,
                        **segment.metadata,
                    },
                )
            )

        combined_text_parts: list[str] = []

        if content.text.strip():
            combined_text_parts.append(
                content.text.strip()
            )

        if result.text.strip():
            combined_text_parts.append(
                result.text.strip()
            )

        combined_text = "\n\n".join(
            combined_text_parts
        )

        return content.model_copy(
            update={
                "text": combined_text,
                "blocks": (
                    existing_blocks
                    + transcript_blocks
                ),
                "language": (
                    content.language
                    or result.language
                ),
                "metadata": metadata,
            }
        )

    @staticmethod
    def _record_asr_resolution_failure(
        content: ExtractedContent,
        error: Exception,
    ) -> ExtractedContent:
        """
        Record ASR source-resolution failure without
        failing the complete ingestion pipeline.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["asr"] = {
            "status": "not_available",
            "reason": (
                "Audio content could not be resolved "
                "for ASR enrichment."
            ),
            "error_type": type(error).__name__,
            "error": str(error),
        }

        return content.model_copy(
            update={
                "metadata": metadata,
            }
        )