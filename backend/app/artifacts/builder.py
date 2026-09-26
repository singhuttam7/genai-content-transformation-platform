from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from app.agents.transformation.contracts import ArtifactEnvelope


class ArtifactBuildData:
    """
    Persistence-ready representation of a generated artifact.

    This is intentionally not a SQLAlchemy model. It is an application-layer
    data object that can later be consumed by the persistence layer.
    """

    def __init__(
        self,
        *,
        transformation_id: UUID,
        execution_id: UUID,
        artifact_type: str,
        title: str | None,
        content: str | None,
        content_hash: str | None,
        metadata: dict[str, Any],
        status: str,
        storage_uri: str | None = None,
    ) -> None:
        self.transformation_id = transformation_id
        self.execution_id = execution_id
        self.artifact_type = artifact_type
        self.title = title
        self.content = content
        self.content_hash = content_hash
        self.metadata = metadata
        self.status = status
        self.storage_uri = storage_uri


class ArtifactBuilder:
    """
    Convert application-level ArtifactEnvelope objects into persistence-ready
    artifact data.

    Persistence itself remains outside this builder.
    """

    def build(
        self,
        *,
        envelope: ArtifactEnvelope,
        transformation_id: UUID,
        execution_id: UUID,
        status: str = "GENERATED",
        storage_uri: str | None = None,
    ) -> ArtifactBuildData:
        if not isinstance(envelope, ArtifactEnvelope):
            raise TypeError(
                "envelope must be an ArtifactEnvelope."
            )

        if not isinstance(transformation_id, UUID):
            raise TypeError(
                "transformation_id must be a UUID."
            )

        if not isinstance(execution_id, UUID):
            raise TypeError(
                "execution_id must be a UUID."
            )

        if not isinstance(status, str) or not status.strip():
            raise ValueError(
                "status must be a non-empty string."
            )

        content = self._normalize_content(
            envelope.content
        )

        content_hash = (
            self._calculate_hash(content)
            if content is not None
            else None
        )

        return ArtifactBuildData(
            transformation_id=transformation_id,
            execution_id=execution_id,
            artifact_type=envelope.artifact_type,
            title=envelope.title,
            content=content,
            content_hash=content_hash,
            metadata=dict(envelope.metadata),
            status=status.strip(),
            storage_uri=storage_uri,
        )

    @staticmethod
    def _normalize_content(
        content: Any,
    ) -> str | None:
        if content is None:
            return None

        if isinstance(content, str):
            return content

        return json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _calculate_hash(
        content: str,
    ) -> str:
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()