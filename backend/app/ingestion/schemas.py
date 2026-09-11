from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class InputType(StrEnum):
    """Supported ingestion input types."""

    TEXT = "text"
    PROMPT = "prompt"
    URL = "url"

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ProcessingStatus(StrEnum):
    """Lifecycle states for ingestion processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentBlockType(StrEnum):
    """Structural types that can appear in extracted content."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    CODE = "code"
    QUOTE = "quote"
    TRANSCRIPT = "transcript"
    UNKNOWN = "unknown"


class SourceReference(BaseModel):
    """Reference to the original ingested source."""

    source_id: UUID | None = None

    source_type: InputType

    title: str | None = None

    filename: str | None = None

    mime_type: str | None = None

    content_hash: str | None = None

    storage_uri: str | None = None


class ContentBlock(BaseModel):
    """
    A normalized structural block extracted from a source.

    Content blocks preserve structural and positional information
    so downstream components do not need to reconstruct the
    original document structure.
    """

    block_type: ContentBlockType

    content: str

    order: int

    page_number: int | None = None

    start_time: float | None = None

    end_time: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class IngestionRequest(BaseModel):
    """
    Unified input boundary for ingestion.

    Small textual content may be supplied inline.

    Large binary inputs should be represented through a storage
    reference rather than placing the binary payload directly
    inside the application request model.
    """

    project_id: UUID | None = None

    source_id: UUID | None = None

    input_type: InputType

    title: str | None = None

    filename: str | None = None

    mime_type: str | None = None

    # ---------------------------------------------------------
    # Inline content
    # ---------------------------------------------------------

    content: str | bytes | None = None

    # ---------------------------------------------------------
    # Remote reference
    # ---------------------------------------------------------

    url: str | None = None

    # ---------------------------------------------------------
    # Stored binary reference
    # ---------------------------------------------------------

    storage_key: str | None = None

    storage_uri: str | None = None

    # ---------------------------------------------------------
    # Flexible contextual metadata
    # ---------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_input_representation(
        self,
    ) -> "IngestionRequest":
        """
        Validate the relationship between input type and
        supplied representation.
        """

        # -----------------------------------------------------
        # Text / prompt must have inline content.
        # -----------------------------------------------------

        if self.input_type in {
            InputType.TEXT,
            InputType.PROMPT,
        }:
            if self.content is None:
                raise ValueError(
                    "Text and prompt inputs require content."
                )

            if isinstance(self.content, str):
                if not self.content.strip():
                    raise ValueError(
                        "Text and prompt content "
                        "cannot be empty."
                    )

            return self

        # -----------------------------------------------------
        # URL must have a URL.
        # -----------------------------------------------------

        if self.input_type == InputType.URL:
            if not self.url:
                raise ValueError(
                    "URL inputs require a URL."
                )

            if not self.url.strip():
                raise ValueError(
                    "URL cannot be empty."
                )

            return self

        # -----------------------------------------------------
        # Binary/document/image/audio/video inputs require
        # either a storage reference or inline bytes.
        #
        # We allow bytes at the internal ingestion boundary.
        # The API layer will later enforce streaming and
        # upload-size constraints.
        # -----------------------------------------------------

        binary_types = {
            InputType.PDF,
            InputType.DOCX,
            InputType.TXT,
            InputType.MARKDOWN,
            InputType.IMAGE,
            InputType.AUDIO,
            InputType.VIDEO,
        }

        if self.input_type in binary_types:
            has_inline_bytes = isinstance(
                self.content,
                bytes,
            )

            has_storage_reference = bool(
                self.storage_key
                or self.storage_uri
            )

            if not (
                has_inline_bytes
                or has_storage_reference
            ):
                raise ValueError(
                    "Binary and document inputs require "
                    "inline bytes or a storage reference."
                )

            return self

        return self


class ExtractedContent(BaseModel):
    """
    Content extracted from a source processor.

    The processor layer is responsible for extraction.
    Structural blocks retain document-level information.
    """

    source: SourceReference

    text: str

    blocks: list[ContentBlock] = Field(
        default_factory=list
    )

    language: str | None = None

    title: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CanonicalContent(BaseModel):
    """
    Unified semantic representation consumed by downstream
    RAG, agents, validation, and artifact generation.

    The canonical representation deliberately preserves
    structural ContentBlock objects instead of flattening
    them into plain strings.
    """

    source: SourceReference

    title: str | None = None

    language: str | None = None

    text: str

    # ---------------------------------------------------------
    # Structured canonical segments
    # ---------------------------------------------------------
    #
    # Each segment retains:
    # - block type
    # - normalized content
    # - deterministic order
    # - page number
    # - timestamps
    # - metadata
    #
    # This is important for PDF, DOCX, image, audio and video
    # ingestion as well as downstream RAG and provenance.
    # ---------------------------------------------------------

    segments: list[ContentBlock] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Semantic fields
    # ---------------------------------------------------------

    entities: list[str] = Field(
        default_factory=list
    )

    topics: list[str] = Field(
        default_factory=list
    )

    claims: list[str] = Field(
        default_factory=list
    )

    keywords: list[str] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Flexible semantic/contextual information
    # ---------------------------------------------------------

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------

    provenance: dict[str, Any] = Field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Additional metadata
    # ---------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )