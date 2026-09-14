from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.ingestion.schemas import InputType
from app.knowledge.normalization.schemas import (
    NormalizedKnowledgeDocument,
)
from app.knowledge.provenance.metadata import (
    EnrichedKnowledgeMetadata,
    SourceMetadata,
)


class SourceMetadataBuilder:
    """
    Deterministically extract source-specific metadata from a
    normalized knowledge document.

    This component:
    - performs no I/O
    - performs no network requests
    - performs no LLM calls
    - performs no OCR/ASR/vision processing
    - does not mutate the input document
    - preserves source metadata supplied by A4
    """

    VERSION = "1.0"

    def build(
        self,
        *,
        document: NormalizedKnowledgeDocument,
    ) -> EnrichedKnowledgeMetadata:
        """
        Build deterministic source-specific metadata.
        """

        source_type = self._source_type(document)

        source_metadata = self._build_source_metadata(
            document=document,
            source_type=source_type,
        )

        return EnrichedKnowledgeMetadata(
            source_metadata=source_metadata,
            metadata=deepcopy(document.metadata),
        )

    def _build_source_metadata(
        self,
        *,
        document: NormalizedKnowledgeDocument,
        source_type: str,
    ) -> SourceMetadata:
        metadata = deepcopy(document.metadata)

        extracted: dict[str, Any] = {
            "builder": "source_metadata",
            "builder_version": self.VERSION,
        }

        if source_type == InputType.PDF.value:
            extracted.update(
                self._extract_pdf_metadata(metadata)
            )

        elif source_type == InputType.DOCX.value:
            extracted.update(
                self._extract_docx_metadata(
                    document=document,
                )
            )

        elif source_type in {
            InputType.AUDIO.value,
            InputType.VIDEO.value,
        }:
            extracted.update(
                self._extract_media_metadata(
                    document=document,
                )
            )

        elif source_type == InputType.URL.value:
            extracted.update(
                self._extract_url_metadata(metadata)
            )

        else:
            extracted.update(
                self._extract_generic_metadata(metadata)
            )

        return SourceMetadata(
            source_type=source_type,
            metadata=extracted,
        )

    @staticmethod
    def _source_type(
        document: NormalizedKnowledgeDocument,
    ) -> str:
        source_type = document.source.source_type

        if hasattr(source_type, "value"):
            return str(source_type.value)

        return str(source_type)

    @staticmethod
    def _extract_pdf_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key in (
            "processor",
            "page_count",
            "extracted_page_count",
        ):
            if key in metadata:
                result[key] = deepcopy(
                    metadata[key]
                )

        return result

    @staticmethod
    def _extract_docx_metadata(
        *,
        document: NormalizedKnowledgeDocument,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        styles: list[str] = []
        heading_levels: list[int] = []
        list_block_count = 0

        for element in document.elements:
            metadata = element.metadata

            style = metadata.get("style")
            if style is not None:
                styles.append(str(style))

            heading_level = metadata.get(
                "heading_level"
            )
            if heading_level is not None:
                try:
                    heading_levels.append(
                        int(heading_level)
                    )
                except (TypeError, ValueError):
                    pass

            if metadata.get("list") is True:
                list_block_count += 1

        if styles:
            result["styles"] = styles

        if heading_levels:
            result["heading_levels"] = (
                heading_levels
            )

        if list_block_count:
            result["list_block_count"] = (
                list_block_count
            )

        return result

    @staticmethod
    def _extract_media_metadata(
        *,
        document: NormalizedKnowledgeDocument,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        media_metadata = document.metadata.get(
            "media"
        )

        if isinstance(media_metadata, dict):
            result["media"] = deepcopy(
                media_metadata
            )

        for key in (
            "duration",
            "codec",
            "sample_rate",
            "channels",
            "bitrate",
            "format",
            "transcription_status",
        ):
            if key in document.metadata:
                result[key] = deepcopy(
                    document.metadata[key]
                )

        return result

    @staticmethod
    def _extract_url_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key in (
            "source_url",
            "url",
            "canonical_url",
            "title",
            "description",
            "author",
            "language",
            "keywords",
            "opengraph",
            "article",
        ):
            if key in metadata:
                result[key] = deepcopy(
                    metadata[key]
                )

        return result

    @staticmethod
    def _extract_generic_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return deepcopy(metadata)