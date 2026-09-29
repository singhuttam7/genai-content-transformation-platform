from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.agents.transformation.contracts import ArtifactEnvelope
from app.api.dependencies import (
    ArtifactPersistenceServiceDependency,
    DatabaseSession,
)
from app.models.artifact import Artifact
from app.models.execution import Execution
from app.models.transformation import Transformation
from app.schemas.artifact import (
    ArtifactCreateRequest,
    ArtifactListResponse,
    ArtifactResponse,
)


router = APIRouter(
    prefix="/artifacts",
    tags=["Artifacts"],
)


def _to_response(
    artifact: Artifact,
) -> ArtifactResponse:
    """
    Convert the SQLAlchemy Artifact model into the public API schema.

    Important:
    SQLAlchemy's declarative base already exposes a `metadata` attribute.
    The Artifact model stores our JSON metadata in `artifact_metadata`,
    so we must explicitly map it instead of relying on from_attributes.
    """
    return ArtifactResponse(
        id=artifact.id,
        transformation_id=artifact.transformation_id,
        execution_id=artifact.execution_id,
        artifact_type=artifact.artifact_type,
        title=artifact.title,
        content=artifact.content,
        storage_uri=artifact.storage_uri,
        content_hash=artifact.content_hash,
        metadata=dict(
            artifact.artifact_metadata or {},
        ),
        status=artifact.status,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


@router.post(
    "",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_artifact(
    request: ArtifactCreateRequest,
    db: DatabaseSession,
    artifact_service: ArtifactPersistenceServiceDependency,
) -> ArtifactResponse:
    """
    Create and persist an artifact.

    The API layer is responsible for:
    - validating referenced resources
    - enforcing execution/transformation ownership
    - constructing the artifact envelope

    ArtifactPersistenceService is responsible for:
    - building the persistence representation
    - normalizing content
    - calculating the content hash
    - creating and persisting the Artifact model
    """

    transformation_result = await db.execute(
        select(Transformation).where(
            Transformation.id
            == request.transformation_id,
        )
    )

    transformation = (
        transformation_result.scalar_one_or_none()
    )

    if transformation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transformation not found.",
        )

    execution_result = await db.execute(
        select(Execution).where(
            Execution.id
            == request.execution_id,
        )
    )

    execution = (
        execution_result.scalar_one_or_none()
    )

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found.",
        )

    if (
        execution.transformation_id
        != transformation.id
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "Execution and transformation must "
                "refer to the same transformation."
            ),
        )

    envelope = ArtifactEnvelope(
        artifact_type=request.artifact_type,
        title=request.title,
        content=request.content,
        metadata=request.metadata,
    )

    artifact = await artifact_service.persist(
        envelope=envelope,
        transformation_id=transformation.id,
        execution_id=execution.id,
        status=request.status,
        storage_uri=request.storage_uri,
    )

    return _to_response(artifact)


@router.get(
    "",
    response_model=ArtifactListResponse,
)
async def list_artifacts(
    db: DatabaseSession,
) -> ArtifactListResponse:
    """List all artifacts ordered newest first."""

    result = await db.execute(
        select(Artifact).order_by(
            Artifact.created_at.desc(),
        )
    )

    artifacts = list(
        result.scalars().all()
    )

    total_result = await db.execute(
        select(func.count()).select_from(
            Artifact,
        )
    )

    total = total_result.scalar_one()

    return ArtifactListResponse(
        items=[
            _to_response(artifact)
            for artifact in artifacts
        ],
        total=total,
    )


@router.get(
    "/{artifact_id}",
    response_model=ArtifactResponse,
)
async def get_artifact(
    artifact_id: UUID,
    db: DatabaseSession,
) -> ArtifactResponse:
    """Return a single artifact by ID."""

    result = await db.execute(
        select(Artifact).where(
            Artifact.id == artifact_id,
        )
    )

    artifact = (
        result.scalar_one_or_none()
    )

    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    return _to_response(artifact)