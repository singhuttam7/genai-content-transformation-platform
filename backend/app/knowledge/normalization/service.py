from __future__ import annotations

import hashlib
from typing import Any

from app.ingestion.schemas import (
    CanonicalContent,
    ContentBlock,
    ContentBlockType,
)

from app.knowledge.normalization.schemas import (
    NormalizedKnowledgeDocument,
    NormalizedKnowledgeElement,
)


class KnowledgeContentNormalizer:
    """
    Deterministic adapter from A4 CanonicalContent to the
    knowledge-layer representation.

    This service does not:
    - call an LLM
    - generate embeddings
    - perform chunking
    - perform retrieval
    - access PostgreSQL
    - access a vector store
    """

    VERSION = "1.0"

    NON_TEXT_BLOCK_TYPES = frozenset(
        {
            ContentBlockType.IMAGE,
            ContentBlockType.AUDIO,
        }
    )

    async def normalize(
        self,
        content: CanonicalContent,
    ) -> NormalizedKnowledgeDocument:
        """
        Convert CanonicalContent into a deterministic
        knowledge representation.
        """

        normalized_text = self._normalize_document_text(
            content.text,
        )

        elements = self._normalize_segments(
            content.segments,
        )

        provenance = dict(
            content.provenance,
        )

        provenance.update(
            {
                "knowledge_normalizer": "default",
                "knowledge_normalization_version": self.VERSION,
            }
        )

        metadata = dict(
            content.metadata,
        )

        metadata.setdefault(
            "knowledge_normalization",
            {},
        )

        metadata["knowledge_normalization"] = {
            "version": self.VERSION,
            "element_count": len(elements),
        }

        content_hash = self._calculate_content_hash(
            normalized_text=normalized_text,
            elements=elements,
        )

        return NormalizedKnowledgeDocument(
            source=content.source,
            title=self._normalize_optional_text(
                content.title,
            ),
            language=self._normalize_language(
                content.language,
            ),
            text=normalized_text,
            elements=elements,
            entities=list(content.entities),
            topics=list(content.topics),
            claims=list(content.claims),
            keywords=list(content.keywords),
            context=dict(content.context),
            provenance=provenance,
            metadata=metadata,
            content_hash=content_hash,
        )

    # =========================================================
    # Document text
    # =========================================================

    @staticmethod
    def _normalize_document_text(
        text: str,
    ) -> str:
        """
        Apply only safe deterministic normalization.

        A4 has already performed primary whitespace
        normalization, so A5 does not collapse meaningful
        internal whitespace.
        """

        if not text:
            return ""

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            line.rstrip()
            for line in text.split("\n")
        ]

        while lines and not lines[0].strip():
            lines.pop(0)

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)

    # =========================================================
    # Structural segments
    # =========================================================

    def _normalize_segments(
        self,
        segments: list[ContentBlock],
    ) -> list[NormalizedKnowledgeElement]:
        """
        Convert CanonicalContent segments into ordered
        knowledge elements.
        """

        normalized: list[NormalizedKnowledgeElement] = []

        for index, segment in enumerate(segments):
            content = self._normalize_segment_text(
                segment.content,
            )

            if (
                not content
                and segment.block_type
                not in self.NON_TEXT_BLOCK_TYPES
            ):
                continue

            metadata = dict(
                segment.metadata,
            )

            metadata.setdefault(
                "canonical_order",
                segment.order,
            )

            normalized.append(
                NormalizedKnowledgeElement(
                    content=content,
                    block_type=segment.block_type,
                    order=index,
                    page_number=segment.page_number,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    metadata=metadata,
                )
            )

        return normalized

    @staticmethod
    def _normalize_segment_text(
        text: str,
    ) -> str:
        """Normalize safe boundary whitespace for a segment."""

        if not text:
            return ""

        return text.strip()

    # =========================================================
    # Metadata
    # =========================================================

    @staticmethod
    def _normalize_optional_text(
        value: str | None,
    ) -> str | None:
        """Normalize an optional textual field."""

        if value is None:
            return None

        normalized = value.strip()

        return normalized or None

    @staticmethod
    def _normalize_language(
        language: str | None,
    ) -> str | None:
        """Normalize language identifier casing."""

        if language is None:
            return None

        normalized = language.strip().lower()

        return normalized or None

    # =========================================================
    # Content hashing
    # =========================================================

    @staticmethod
    def _calculate_content_hash(
        *,
        normalized_text: str,
        elements: list[NormalizedKnowledgeElement],
    ) -> str:
        """
        Calculate a deterministic SHA-256 hash over the
        normalized knowledge representation.
        """

        hasher = hashlib.sha256()

        hasher.update(
            normalized_text.encode(
                "utf-8",
            )
        )

        for element in elements:
            hasher.update(
                b"\x00",
            )

            hasher.update(
                element.block_type.value.encode(
                    "utf-8",
                )
            )

            hasher.update(
                b"\x00",
            )

            hasher.update(
                str(element.order).encode(
                    "utf-8",
                )
            )

            hasher.update(
                b"\x00",
            )

            hasher.update(
                element.content.encode(
                    "utf-8",
                )
            )

            if element.page_number is not None:
                hasher.update(
                    b"\x00page:",
                )

                hasher.update(
                    str(element.page_number).encode(
                        "utf-8",
                    )
                )

            if element.start_time is not None:
                hasher.update(
                    b"\x00start:",
                )

                hasher.update(
                    str(element.start_time).encode(
                        "utf-8",
                    )
                )

            if element.end_time is not None:
                hasher.update(
                    b"\x00end:",
                )

                hasher.update(
                    str(element.end_time).encode(
                        "utf-8",
                    )
                )

        return hasher.hexdigest()