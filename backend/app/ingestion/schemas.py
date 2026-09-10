from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InputType(StrEnum):
    """Supported categories of source input."""

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
    """Types of structured content blocks."""

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
    """Reference to the source being processed."""

    model_config = ConfigDict(extra="forbid")

    source_id: UUID | None = None
    source_type: InputType
    title: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    content_hash: str | None = None


class ContentBlock(BaseModel):
    """A structured unit extracted from source content."""

    model_config = ConfigDict(extra="allow")

    block_type: ContentBlockType
    content: str = ""
    order: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionRequest(BaseModel):
    """Normalized request entering the ingestion pipeline."""

    model_config = ConfigDict(extra="allow")

    project_id: UUID | None = None
    source_id: UUID | None = None

    input_type: InputType

    title: str | None = None
    filename: str | None = None
    mime_type: str | None = None

    content: str | None = None
    storage_uri: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedContent(BaseModel):
    """Content extracted from a source before canonicalization."""

    model_config = ConfigDict(extra="allow")

    source: SourceReference

    text: str = ""
    blocks: list[ContentBlock] = Field(default_factory=list)

    language: str | None = None
    title: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalContent(BaseModel):
    """Unified representation consumed by downstream AI systems."""

    model_config = ConfigDict(extra="allow")

    source: SourceReference

    title: str | None = None
    language: str | None = None

    text: str = ""
    segments: list[ContentBlock] = Field(default_factory=list)

    entities: list[dict[str, Any]] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    context: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)