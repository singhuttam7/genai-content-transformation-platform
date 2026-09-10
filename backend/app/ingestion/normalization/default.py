from __future__ import annotations

import re

from app.ingestion.normalizer import ContentNormalizer
from app.ingestion.schemas import CanonicalContent, ExtractedContent


class DefaultContentNormalizer(ContentNormalizer):
    """Deterministically normalize extracted content into canonical form."""

    async def normalize(
        self,
        content: ExtractedContent,
    ) -> CanonicalContent:
        text = self._normalize_text(content.text)

        if not text:
            raise ValueError("Cannot normalize empty content.")

        # Preserve the processor-produced structure while ensuring
        # deterministic ordering.
        segments = sorted(
            content.blocks,
            key=lambda block: block.order,
        )

        # Normalize block content without changing its semantic meaning.
        normalized_segments = []

        for index, block in enumerate(segments):
            normalized_block = block.model_copy(
                update={
                    "content": self._normalize_text(block.content),
                    "order": index,
                }
            )

            if normalized_block.content:
                normalized_segments.append(normalized_block)

        return CanonicalContent(
            source=content.source,
            title=content.title,
            language=content.language,
            text=text,
            segments=normalized_segments,
            entities=[],
            topics=[],
            claims=[],
            keywords=[],
            context={},
            provenance={
                "ingestion_normalizer": "default",
                "normalization_version": "1.0",
            },
            metadata=content.metadata,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace while preserving paragraph boundaries."""

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove trailing whitespace from each line.
        lines = [line.rstrip() for line in text.split("\n")]

        # Collapse excessive blank lines.
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        # Remove unnecessary surrounding whitespace.
        return normalized.strip()