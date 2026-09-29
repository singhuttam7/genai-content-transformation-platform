from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.transformation.contracts import ArtifactEnvelope
from app.artifacts.builder import ArtifactBuilder
from app.models.artifact import Artifact


class ArtifactPersistenceService:
    """
    Persist generated ArtifactEnvelope objects as database artifacts.

    Responsibilities:
    - validate and build persistence-ready artifact data;
    - convert ArtifactBuildData into Artifact ORM models;
    - persist multiple artifacts in one database transaction.

    This service does not execute transformations or workflows.
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        builder: ArtifactBuilder | None = None,
    ) -> None:
        if not isinstance(session, AsyncSession):
            raise TypeError(
                "session must be an AsyncSession."
            )

        self.session = session
        self.builder = builder or ArtifactBuilder()

    async def persist(
        self,
        *,
        envelope: ArtifactEnvelope,
        transformation_id: UUID,
        execution_id: UUID,
        status: str = "GENERATED",
        storage_uri: str | None = None,
    ) -> Artifact:
        """
        Build and persist one artifact.
        """

        build_data = self.builder.build(
            envelope=envelope,
            transformation_id=transformation_id,
            execution_id=execution_id,
            status=status,
            storage_uri=storage_uri,
        )

        artifact = Artifact(
            transformation_id=build_data.transformation_id,
            execution_id=build_data.execution_id,
            artifact_type=build_data.artifact_type,
            title=build_data.title,
            content=build_data.content,
            storage_uri=build_data.storage_uri,
            content_hash=build_data.content_hash,
            artifact_metadata=dict(
                build_data.metadata,
            ),
            status=build_data.status,
        )

        self.session.add(artifact)

        await self.session.commit()
        await self.session.refresh(artifact)

        return artifact

    async def persist_many(
        self,
        *,
        envelopes: Sequence[ArtifactEnvelope],
        transformation_id: UUID,
        execution_id: UUID,
        status: str = "GENERATED",
    ) -> list[Artifact]:
        """
        Build and persist multiple artifacts atomically.

        All envelopes are validated and converted before anything is
        added to the database. This prevents a malformed later envelope
        from causing a partial persistence operation.
        """

        build_data_items = [
            self.builder.build(
                envelope=envelope,
                transformation_id=transformation_id,
                execution_id=execution_id,
                status=status,
            )
            for envelope in envelopes
        ]

        artifacts = [
            Artifact(
                transformation_id=build_data.transformation_id,
                execution_id=build_data.execution_id,
                artifact_type=build_data.artifact_type,
                title=build_data.title,
                content=build_data.content,
                storage_uri=build_data.storage_uri,
                content_hash=build_data.content_hash,
                artifact_metadata=dict(
                    build_data.metadata,
                ),
                status=build_data.status,
            )
            for build_data in build_data_items
        ]

        self.session.add_all(artifacts)

        await self.session.commit()

        for artifact in artifacts:
            await self.session.refresh(artifact)

        return artifacts