from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StorageObjectType(StrEnum):
    """Logical categories of objects stored by the platform."""

    SOURCE_FILE = "source_file"
    ARTIFACT = "artifact"
    INTERMEDIATE = "intermediate"


class StorageObject(BaseModel):
    """Metadata describing a stored object."""

    model_config = ConfigDict(extra="forbid")

    object_id: UUID
    object_type: StorageObjectType
    storage_key: str
    uri: str
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int = Field(ge=0)
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StorageUpload(BaseModel):
    """Description of content that should be stored."""

    model_config = ConfigDict(extra="forbid")

    object_id: UUID
    object_type: StorageObjectType
    filename: str
    content_type: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)