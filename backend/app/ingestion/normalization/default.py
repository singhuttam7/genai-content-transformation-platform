from __future__ import annotations

import re

from app.ingestion.schemas import (
    CanonicalContent,
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
)


class DefaultContentNormalizer:
    """
    Deterministic content normalizer.

    Responsibilities:
    - Normalize extracted text.
    - Normalize structural content blocks.
    - Preserve structural information in canonical segments.
    - Preserve source metadata.
    - Attach normalization provenance.

    This component intentionally performs deterministic
    normalization only.

    Semantic understanding will be handled by the
    content-understanding layer later.
    """

    VERSION = "1.0"

    # ---------------------------------------------------------
    # Structural blocks that are valid even when they do not
    # contain textual content.
    #
    # These represent source-level multimodal content rather
    # than text-bearing content.
    # ---------------------------------------------------------

    NON_TEXT_BLOCK_TYPES = frozenset(
        {
            ContentBlockType.IMAGE,
        }
    )

    async def normalize(
        self,
        content: ExtractedContent,
    ) -> CanonicalContent:
        """
        Convert ExtractedContent into CanonicalContent.
        """

        # =====================================================
        # 1. Normalize main text
        # =====================================================

        normalized_text = self._normalize_text(
            content.text
        )

        # =====================================================
        # 2. Normalize and order structural blocks
        # =====================================================

        normalized_blocks = self._normalize_blocks(
            content.blocks,
            normalized_text=normalized_text,
        )

        # =====================================================
        # 3. Preserve structured blocks as canonical segments
        # =====================================================

        segments = normalized_blocks

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
        Normalize document-level whitespace while preserving
        meaningful internal indentation.

        Rules:
        - Normalize CRLF/CR to LF.
        - Remove trailing whitespace from each line.
        - Collapse consecutive blank lines to one blank line.
        - Remove leading blank lines.
        - Remove trailing blank lines.
        - Remove leading whitespace from the beginning of the
          entire document.
        - Remove trailing whitespace from the end of the
          entire document.
        """

        if not text:
            return ""

        # -----------------------------------------------------
        # Normalize line endings.
        # -----------------------------------------------------

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # -----------------------------------------------------
        # Remove trailing whitespace from each line.
        #
        # Do NOT call strip() here because leading indentation
        # inside the document must be preserved.
        # -----------------------------------------------------

        lines = [
            line.rstrip()
            for line in text.split("\n")
        ]

        normalized_lines: list[str] = []

        previous_blank = False

        for line in lines:
            # -------------------------------------------------
            # Determine whether the line is blank without
            # modifying the actual line content.
            # -------------------------------------------------

            if not line.strip():
                if not previous_blank:
                    normalized_lines.append("")

                previous_blank = True

                continue

            # -------------------------------------------------
            # Preserve leading whitespace on internal lines.
            # -------------------------------------------------

            normalized_lines.append(line)

            previous_blank = False

        # -----------------------------------------------------
        # Remove leading blank lines.
        # -----------------------------------------------------

        while (
            normalized_lines
            and not normalized_lines[0]
        ):
            normalized_lines.pop(0)

        # -----------------------------------------------------
        # Remove trailing blank lines.
        # -----------------------------------------------------

        while (
            normalized_lines
            and not normalized_lines[-1]
        ):
            normalized_lines.pop()

        if not normalized_lines:
            return ""

        # -----------------------------------------------------
        # Reconstruct normalized document.
        # -----------------------------------------------------

        normalized = "\n".join(
            normalized_lines
        )

        # -----------------------------------------------------
        # Remove whitespace only from the outer boundaries of
        # the complete document.
        #
        # Internal indentation remains untouched.
        # -----------------------------------------------------

        normalized = normalized.lstrip()

        normalized = normalized.rstrip()

        return normalized

    # =========================================================
    # Block normalization
    # =========================================================

    @classmethod
    def _normalize_blocks(
        cls,
        blocks: list[ContentBlock],
        *,
        normalized_text: str | None = None,
    ) -> list[ContentBlock]:
        """
        Normalize structural blocks and guarantee deterministic
        ordering.

        Original processor order values are used for sorting.

        New order values are reassigned sequentially after
        normalization so the canonical representation always
        has deterministic zero-based ordering.

        When exactly one block represents the complete source,
        its normalized content is aligned with normalized_text.

        Non-text structural blocks such as IMAGE are preserved
        even when their textual content is empty.
        """

        normalized: list[ContentBlock] = []

        # -----------------------------------------------------
        # Sort according to processor-provided order.
        # -----------------------------------------------------

        sorted_blocks = sorted(
            blocks,
            key=lambda block: block.order,
        )

        # -----------------------------------------------------
        # Normalize each block.
        # -----------------------------------------------------

        for block in sorted_blocks:
            # -------------------------------------------------
            # If there is exactly one block and it represents
            # the complete source, use the canonical normalized
            # document text.
            #
            # This keeps:
            #
            #     CanonicalContent.text
            #
            # and:
            #
            #     CanonicalContent.segments[0].content
            #
            # consistent for TEXT/PROMPT inputs.
            #
            # Do not apply this rule to non-text blocks because
            # their source representation may legitimately have
            # empty textual content.
            # -------------------------------------------------

            if (
                normalized_text is not None
                and len(sorted_blocks) == 1
                and block.block_type
                not in cls.NON_TEXT_BLOCK_TYPES
            ):
                normalized_content = normalized_text

            else:
                normalized_content = (
                    cls._normalize_block_content(
                        block.content
                    )
                )

            # -------------------------------------------------
            # Ignore empty text-bearing blocks.
            #
            # Non-text structural blocks such as IMAGE are
            # preserved because their metadata can represent
            # meaningful source information even without OCR
            # or vision processing.
            # -------------------------------------------------

            if (
                not normalized_content
                and block.block_type
                not in cls.NON_TEXT_BLOCK_TYPES
            ):
                continue

            normalized.append(
                block.model_copy(
                    update={
                        "content": normalized_content,
                    }
                )
            )

        # -----------------------------------------------------
        # Reassign deterministic canonical ordering.
        # -----------------------------------------------------

        return [
            block.model_copy(
                update={
                    "order": index,
                }
            )
            for index, block in enumerate(
                normalized
            )
        ]

    # =========================================================
    # Individual block normalization
    # =========================================================

    @staticmethod
    def _normalize_block_content(
        text: str,
    ) -> str:
        """
        Normalize whitespace inside an individual structural
        block.
        """

        if not text:
            return ""

        # -----------------------------------------------------
        # Normalize line endings.
        # -----------------------------------------------------

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # -----------------------------------------------------
        # Remove trailing whitespace from each line.
        # -----------------------------------------------------

        text = "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

        # -----------------------------------------------------
        # Collapse excessive horizontal whitespace.
        # -----------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text.strip()