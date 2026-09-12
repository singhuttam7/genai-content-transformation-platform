from __future__ import annotations

from copy import deepcopy
from typing import Any

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

from app.ingestion.video.asr import (
    VideoASRService,
)
from app.ingestion.video.schemas import (
    VideoASRRequest,
    VideoASRStatus,
    VideoInfo,
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
    - Invoke video-specific ASR processing.
    - Preserve enrichment metadata.
    - Treat enrichment failures as non-fatal where appropriate.

    Current enrichment capabilities:

        IMAGE -> OCR
        AUDIO -> ASR
        VIDEO -> VideoASRService
    """

    def __init__(
        self,
        *,
        content_resolver: InputContentResolver,
        ocr_provider: OCRProvider | None = None,
        asr_provider: ASRProvider | None = None,
        video_asr_service: VideoASRService | None = None,
    ) -> None:
        self.content_resolver = content_resolver
        self.ocr_provider = ocr_provider
        self.asr_provider = asr_provider
        self.video_asr_service = video_asr_service

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
        VIDEO -> Video ASR
        Other input types -> unchanged
        """

        enriched = content.model_copy(
            deep=True
        )

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

        if request.input_type == InputType.VIDEO:
            return await self._enrich_video(
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

    # =========================================================
    # VIDEO -> ASR
    # =========================================================

    async def _enrich_video(
        self,
        *,
        request: IngestionRequest,
        content: ExtractedContent,
    ) -> ExtractedContent:
        """
        Perform video-to-ASR enrichment.

        The video processor is responsible for inspecting the
        video and placing provider-independent VideoInfo inside:

            content.metadata["video"]["video_info"]

        This method:
        1. Resolves the original video bytes.
        2. Retrieves VideoInfo.
        3. Builds VideoASRRequest.
        4. Invokes VideoASRService.
        5. Preserves the structured video-ASR result.
        6. Reuses the existing ASRResult -> TRANSCRIPT mapping.

        Video ASR failures do not discard the original video
        content.
        """

        if self.video_asr_service is None:
            return content

        # ---------------------------------------------------------
        # 1. Resolve original video bytes
        # ---------------------------------------------------------

        try:
            video_bytes = await self.content_resolver.resolve(
                request
            )

        except Exception as exc:
            return self._record_video_asr_resolution_failure(
                content,
                exc,
            )

        # ---------------------------------------------------------
        # 2. Retrieve VideoInfo produced by the video processor
        # ---------------------------------------------------------

        video_info = self._get_video_info(
            content
        )

        if video_info is None:
            return self._record_video_asr_failure(
                content=content,
                status=VideoASRStatus.ASR_FAILED,
                reason=(
                    "VideoInfo was not available from "
                    "the video processing stage."
                ),
                error_type="MissingVideoInfo",
            )

        # ---------------------------------------------------------
        # 3. Build VideoASRRequest
        # ---------------------------------------------------------

        video_asr_request = VideoASRRequest(
            language=self._get_video_asr_language(
                request
            ),
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

        # ---------------------------------------------------------
        # 4. Invoke VideoASRService
        # ---------------------------------------------------------

        try:
            video_asr_result = (
                await self.video_asr_service.transcribe(
                    video_bytes,
                    video_info=video_info,
                    request=video_asr_request,
                    filename=request.filename,
                )
            )

        except Exception as exc:
            return self._record_video_asr_failure(
                content=content,
                status=VideoASRStatus.ASR_FAILED,
                reason=(
                    "Video ASR processing failed."
                ),
                error_type=type(exc).__name__,
                error=str(exc),
            )

        # ---------------------------------------------------------
        # 5. Preserve structured video ASR metadata
        # ---------------------------------------------------------

        enriched = self._apply_video_asr_metadata(
            content=content,
            result=video_asr_result,
        )

        # ---------------------------------------------------------
        # 6. Reuse existing ASR result mapper
        # ---------------------------------------------------------

        if (
            video_asr_result.asr_result is None
        ):
            return enriched

        return self._apply_asr_result(
            content=enriched,
            result=video_asr_result.asr_result,
        )

    # =========================================================
    # VIDEO metadata helpers
    # =========================================================

    @staticmethod
    def _get_video_info(
        content: ExtractedContent,
    ) -> VideoInfo | None:
        """
        Retrieve VideoInfo from the structured metadata emitted
        by VideoDocumentProcessor.

        Returns None if the metadata is missing or malformed.
        """

        video_metadata = content.metadata.get(
            "video"
        )

        if not isinstance(
            video_metadata,
            dict,
        ):
            return None

        raw_video_info = video_metadata.get(
            "video_info"
        )

        if isinstance(
            raw_video_info,
            VideoInfo,
        ):
            return raw_video_info

        if not isinstance(
            raw_video_info,
            dict,
        ):
            return None

        try:
            return VideoInfo.model_validate(
                raw_video_info
            )
        except Exception:
            return None

    @staticmethod
    def _get_video_asr_language(
        request: IngestionRequest,
    ) -> str | None:
        """
        Resolve the requested video-ASR language.

        The video-specific key is preferred, with the general ASR
        key retained as a compatibility fallback.
        """

        language = request.metadata.get(
            "video_asr_language"
        )

        if language is None:
            language = request.metadata.get(
                "asr_language"
            )

        if language is None:
            return None

        return str(language)

    @staticmethod
    def _apply_video_asr_metadata(
        *,
        content: ExtractedContent,
        result: Any,
    ) -> ExtractedContent:
        """
        Preserve the complete structured VideoASRResult metadata.

        This keeps capability-level information such as:

        - no audio
        - audio extraction failure
        - ASR failure
        - no speech
        - provider metadata
        - extraction metadata

        separate from the canonical transcript blocks.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["video_asr"] = {
            "status": result.status.value,
            "audio_extraction": (
                deepcopy(
                    result.audio_extraction
                )
            ),
            "errors": list(
                result.errors
            ),
            **deepcopy(
                result.metadata
            ),
        }

        return content.model_copy(
            update={
                "metadata": metadata,
            }
        )

    @staticmethod
    def _record_video_asr_resolution_failure(
        content: ExtractedContent,
        error: Exception,
    ) -> ExtractedContent:
        """
        Record video source-resolution failure without
        discarding the original extracted video content.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["video_asr"] = {
            "status": (
                VideoASRStatus.ASR_FAILED.value
            ),
            "stage": "video_resolution",
            "error_type": type(error).__name__,
            "error": str(error),
            "retryable": False,
        }

        return content.model_copy(
            update={
                "metadata": metadata,
            }
        )

    @staticmethod
    def _record_video_asr_failure(
        *,
        content: ExtractedContent,
        status: VideoASRStatus,
        reason: str,
        error_type: str,
        error: str | None = None,
    ) -> ExtractedContent:
        """
        Record video ASR failure without destroying the original
        extracted content.
        """

        metadata = deepcopy(
            content.metadata
        )

        failure_metadata: dict[str, object] = {
            "status": status.value,
            "stage": "video_asr",
            "reason": reason,
            "error_type": error_type,
            "retryable": False,
        }

        if error is not None:
            failure_metadata["error"] = error

        metadata["video_asr"] = failure_metadata

        return content.model_copy(
            update={
                "metadata": metadata,
            }
        )

    # =========================================================
    # Existing ASR result mapping
    # =========================================================

    @staticmethod
    def _apply_asr_result(
        *,
        content: ExtractedContent,
        result: ASRResult,
    ) -> ExtractedContent:
        """
        Apply ASR result to extracted content.

        Each ASR segment becomes a TRANSCRIPT block containing
        timing and provider metadata.

        This method is intentionally shared by:

            AUDIO -> ASR
            VIDEO -> VideoASR -> ASRResult
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["asr"] = {
            "status": result.status.value,
            "provider": result.provider,
            "language": result.language,
            "confidence": result.confidence,
            "segment_count": len(
                result.segments
            ),
            **result.metadata,
        }

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

        transcript_blocks: list[
            ContentBlock
        ] = []

        for offset, segment in enumerate(
            result.segments
        ):
            transcript_blocks.append(
                ContentBlock(
                    block_type=(
                        ContentBlockType.TRANSCRIPT
                    ),
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