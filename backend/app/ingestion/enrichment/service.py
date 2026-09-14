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
    VisionRequest,
    VisionResult,
    VisionStatus,
)

from app.ingestion.video.vision_orchestration import (
    VideoVisionService,
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
    - Invoke video-specific visual analysis.
    - Preserve enrichment metadata.
    - Treat enrichment failures as non-fatal where appropriate.

    Current enrichment capabilities:

        IMAGE -> OCR
        AUDIO -> ASR
        VIDEO -> VideoASRService
        VIDEO -> VideoVisionService
    """

    def __init__(
        self,
        *,
        content_resolver: InputContentResolver,
        ocr_provider: OCRProvider | None = None,
        asr_provider: ASRProvider | None = None,
        video_asr_service: VideoASRService | None = None,
        video_vision_service: VideoVisionService | None = None,
    ) -> None:
        self.content_resolver = content_resolver
        self.ocr_provider = ocr_provider
        self.asr_provider = asr_provider
        self.video_asr_service = video_asr_service
        self.video_vision_service = video_vision_service

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
        VIDEO -> Video ASR + Video Vision
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
    # VIDEO -> ASR + VISION
    # =========================================================

    async def _enrich_video(
        self,
        *,
        request: IngestionRequest,
        content: ExtractedContent,
    ) -> ExtractedContent:
        """
        Perform video enrichment.

        Video enrichment consists of two independent capabilities:

            Video -> Video ASR
            Video -> Video Vision

        Each capability is isolated so that failure of one
        capability does not destroy successful results from
        the other capability.
        """

        enriched = content

        if self.video_asr_service is not None:
            enriched = await self._enrich_video_asr(
                request=request,
                content=enriched,
            )

        if self.video_vision_service is not None:
            enriched = await self._enrich_video_vision(
                request=request,
                content=enriched,
            )

        return enriched

    # =========================================================
    # VIDEO -> ASR
    # =========================================================

    async def _enrich_video_asr(
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

        Video ASR failures do not discard the original video
        content.
        """

        try:
            video_bytes = await self.content_resolver.resolve(
                request
            )

        except Exception as exc:
            return self._record_video_asr_resolution_failure(
                content,
                exc,
            )

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

        enriched = self._apply_video_asr_metadata(
            content=content,
            result=video_asr_result,
        )

        if (
            video_asr_result.asr_result is None
        ):
            return enriched

        return self._apply_asr_result(
            content=enriched,
            result=video_asr_result.asr_result,
        )

    # =========================================================
    # VIDEO -> VISION
    # =========================================================

    async def _enrich_video_vision(
        self,
        *,
        request: IngestionRequest,
        content: ExtractedContent,
    ) -> ExtractedContent:
        """
        Perform visual analysis over extracted video frames.

        The workflow is:

            video bytes
                ->
            VideoInfo
                ->
            VideoVisionService
                ->
            VisionResult
                ->
            visual ContentBlocks

        Visual analysis is non-fatal. The original VIDEO block
        and all previously generated enrichment remain intact
        when vision processing fails.
        """

        try:
            video_bytes = await self.content_resolver.resolve(
                request
            )

        except Exception as exc:
            return self._record_video_vision_resolution_failure(
                content,
                exc,
            )

        video_info = self._get_video_info(
            content
        )

        if video_info is None:
            return self._record_video_vision_failure(
                content=content,
                status=VisionStatus.FAILED,
                reason=(
                    "VideoInfo was not available from "
                    "the video processing stage."
                ),
                error_type="MissingVideoInfo",
            )

        vision_request = self._build_video_vision_request(
            request
        )

        try:
            vision_result = (
                await self.video_vision_service.analyze(
                    video_bytes,
                    video_info=video_info,
                    vision_request=vision_request,
                )
            )

        except Exception as exc:
            return self._record_video_vision_failure(
                content=content,
                status=VisionStatus.FAILED,
                reason=(
                    "Video vision processing failed."
                ),
                error_type=type(exc).__name__,
                error=str(exc),
            )

        enriched = self._apply_video_vision_metadata(
            content=content,
            result=vision_result,
        )

        return self._apply_video_vision_observations(
            content=enriched,
            result=vision_result,
        )

    @staticmethod
    def _build_video_vision_request(
        request: IngestionRequest,
    ) -> VisionRequest:
        """
        Build the provider-independent vision request.

        The request metadata supports optional operator controls
        without coupling the enrichment layer to a specific
        vision provider.
        """

        prompt = request.metadata.get(
            "vision_prompt"
        )

        if prompt is not None:
            prompt = str(prompt).strip()

            if not prompt:
                prompt = None

        detail_level = request.metadata.get(
            "vision_detail_level",
            "standard",
        )

        if detail_level is None:
            detail_level = "standard"

        detail_level = str(
            detail_level
        )

        max_observations = request.metadata.get(
            "vision_max_observations"
        )

        if max_observations is not None:
            try:
                max_observations = int(
                    max_observations
                )
            except (
                TypeError,
                ValueError,
            ):
                max_observations = None

        return VisionRequest(
            prompt=prompt,
            detail_level=detail_level,
            max_observations=max_observations,
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
    def _apply_video_vision_metadata(
        *,
        content: ExtractedContent,
        result: VisionResult,
    ) -> ExtractedContent:
        """
        Preserve structured video-vision execution metadata.

        Provider and orchestration metadata are retained rather
        than flattened into the canonical text.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["video_vision"] = {
            "status": result.status.value,
            "requested_frames": (
                result.requested_frames
            ),
            "processed_frames": (
                result.processed_frames
            ),
            "failed_frames": (
                result.failed_frames
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
    def _apply_video_vision_observations(
        *,
        content: ExtractedContent,
        result: VisionResult,
    ) -> ExtractedContent:
        """
        Convert successful visual observations into textual
        ContentBlock objects.

        The original VIDEO block is preserved.

        Each observation retains:
        - timestamp
        - frame index
        - objects
        - entities
        - actions
        - scene
        - visible text
        - confidence
        - provider metadata
        """

        if not result.observations:
            return content

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

        vision_blocks: list[
            ContentBlock
        ] = []

        provider = result.metadata.get(
            "provider"
        )

        for offset, observation in enumerate(
            result.observations
        ):
            description = (
                observation.description
                or ""
            ).strip()

            if not description:
                fallback_parts: list[str] = []

                if observation.scene:
                    fallback_parts.append(
                        observation.scene
                    )

                if observation.visible_text:
                    fallback_parts.append(
                        (
                            "Visible text: "
                            + observation.visible_text
                        )
                    )

                description = " ".join(
                    fallback_parts
                ).strip()

            vision_blocks.append(
                ContentBlock(
                    block_type=(
                        ContentBlockType.PARAGRAPH
                    ),
                    content=description,
                    order=next_order + offset,
                    start_time=(
                        observation.timestamp_seconds
                    ),
                    end_time=(
                        observation.timestamp_seconds
                    ),
                    metadata={
                        "source": "vision",
                        "provider": provider,
                        "frame_index": (
                            observation.frame_index
                        ),
                        "timestamp_seconds": (
                            observation.timestamp_seconds
                        ),
                        "objects": list(
                            observation.objects
                        ),
                        "entities": list(
                            observation.entities
                        ),
                        "actions": list(
                            observation.actions
                        ),
                        "scene": observation.scene,
                        "visible_text": (
                            observation.visible_text
                        ),
                        "confidence": (
                            observation.confidence
                        ),
                        **deepcopy(
                            observation.metadata
                        ),
                    },
                )
            )

        combined_text_parts: list[str] = []

        if content.text.strip():
            combined_text_parts.append(
                content.text.strip()
            )

        observation_text = "\n\n".join(
            block.content
            for block in vision_blocks
            if block.content.strip()
        )

        if observation_text:
            combined_text_parts.append(
                observation_text
            )

        combined_text = "\n\n".join(
            combined_text_parts
        )

        return content.model_copy(
            update={
                "text": combined_text,
                "blocks": (
                    existing_blocks
                    + vision_blocks
                ),
            }
        )

    @staticmethod
    def _record_video_vision_resolution_failure(
        content: ExtractedContent,
        error: Exception,
    ) -> ExtractedContent:
        """
        Record video source-resolution failure without
        discarding the original extracted content.
        """

        metadata = deepcopy(
            content.metadata
        )

        metadata["video_vision"] = {
            "status": "not_available",
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
    def _record_video_vision_failure(
        *,
        content: ExtractedContent,
        status: VisionStatus,
        reason: str,
        error_type: str,
        error: str | None = None,
    ) -> ExtractedContent:
        """
        Record video vision failure without destroying
        previously extracted or enriched content.
        """

        metadata = deepcopy(
            content.metadata
        )

        failure_metadata: dict[str, object] = {
            "status": status.value,
            "stage": "video_vision",
            "reason": reason,
            "error_type": error_type,
            "retryable": False,
        }

        if error is not None:
            failure_metadata["error"] = error

        metadata["video_vision"] = failure_metadata

        return content.model_copy(
            update={
                "metadata": metadata,
            }
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