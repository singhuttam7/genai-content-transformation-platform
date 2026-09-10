from __future__ import annotations

import re

from app.ingestion.schemas import (
    CanonicalContent,
    ContentBlock,
    ExtractedContent,
)


class DefaultContentNormalizer:
    """
    Deterministic content normalizer.

    Responsibilities:
    - Normalize extracted text.
    - Normalize structural content blocks.
    - Build semantic-ready segments.
    - Preserve source metadata.
    - Attach normalization provenance.

    This component intentionally performs deterministic
    normalization only. Semantic understanding will be handled
    by the content-understanding layer later.
    """

    VERSION = "1.0"

    async def normalize(
        self,
        content: ExtractedContent,
    ) -> CanonicalContent:
        """
        Convert extracted content into CanonicalContent.
        """

        # =====================================================
        # 1. Normalize main text
        # =====================================================

        normalized_text = self._normalize_text(
            content.text
        )

        # =====================================================
        # 2. Normalize and order blocks
        # =====================================================

        normalized_blocks = self._normalize_blocks(
            content.blocks
        )

        # =====================================================
        # 3. Build string-based segments
        #
        # CanonicalContent.segments expects:
        #
        #     list[str]
        #
        # not:
        #
        #     list[ContentBlock]
        # =====================================================

        segments = [
            block.content
            for block in normalized_blocks
            if block.content.strip()
        ]

        # =====================================================
        # 4. Build provenance information
        # =====================================================

        provenance = {
            "ingestion_normalizer": "default",
            "normalization_version": self.VERSION,
        }

        # =====================================================
        # 5. Return canonical representation
        # =====================================================

        return CanonicalContent(
            source=content.source,
            title=content.title,
            language=content.language,
            text=normalized_text,
            segments=segments,
            entities=[],
            topics=[],
            claims=[],
            keywords=[],
            context={},
            provenance=provenance,
            metadata=content.metadata,
        )

    # =========================================================
    # Text normalization
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace while preserving paragraph
        boundaries.
        """

        if not text:
            return ""

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Remove trailing whitespace from each line.
        lines = [
            line.rstrip()
            for line in text.split("\n")
        ]

        normalized_lines: list[str] = []

        previous_blank = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if not previous_blank:
                    normalized_lines.append("")

                previous_blank = True
                continue

            normalized_lines.append(
                stripped
            )

            previous_blank = False

        # Remove leading/trailing blank lines.
        while (
            normalized_lines
            and not normalized_lines[0]
        ):
            normalized_lines.pop(0)

        while (
            normalized_lines
            and not normalized_lines[-1]
        ):
            normalized_lines.pop()

        return "\n".join(
            normalized_lines
        )

    # =========================================================
    # Block normalization
    # =========================================================

    @classmethod
    def _normalize_blocks(
        cls,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """
        Normalize structural blocks and guarantee deterministic
        ordering.
        """

        normalized: list[ContentBlock] = []

        sorted_blocks = sorted(
            blocks,
            key=lambda block: block.order,
        )

        for index, block in enumerate(
            sorted_blocks
        ):
            normalized_content = cls._normalize_block_content(
                block.content
            )

            if not normalized_content:
                continue

            normalized.append(
                block.model_copy(
                    update={
                        "content": normalized_content,
                        "order": index,
                    }
                )
            )

        return normalized

    # =========================================================
    # Individual block normalization
    # =========================================================

    @staticmethod
    def _normalize_block_content(
        text: str,
    ) -> str:
        """
        Normalize whitespace inside an individual block.
        """

        if not text:
            return ""

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Remove trailing whitespace.
        text = "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

        # Collapse excessive horizontal whitespace.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text.strip()