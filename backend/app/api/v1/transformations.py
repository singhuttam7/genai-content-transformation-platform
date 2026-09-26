from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.api.dependencies import DatabaseSession
from app.models.transformation import Transformation
from app.schemas.transformation import (
    TransformationCreateRequest,
    TransformationListResponse,
    TransformationResponse,
)


router = APIRouter(
    prefix="/transformations",
    tags=["Transformations"],
)


@router.post(
    "",
    response_model=TransformationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_transformation(
    request: TransformationCreateRequest,
    db: DatabaseSession,
) -> Transformation:
    """Create a new transformation request."""

    transformation = Transformation(
        project_id=request.project_id,
        source_id=request.source_id,
        objective=request.objective,
        audience=request.audience,
        tone=request.tone,
        language=request.language,
        detail_level=request.detail_level,
        style=request.style,
        requested_outputs=request.requested_outputs,
        configuration=request.configuration,
        status="DRAFT",
    )

    db.add(transformation)
    await db.commit()
    await db.refresh(transformation)

    return transformation


@router.get(
    "",
    response_model=TransformationListResponse,
)
async def list_transformations(
    db: DatabaseSession,
) -> TransformationListResponse:
    """List all transformation requests."""

    result = await db.execute(
        select(Transformation).order_by(
            Transformation.created_at.desc()
        )
    )

    items = list(result.scalars().all())

    total_result = await db.execute(
        select(func.count()).select_from(Transformation)
    )

    total = total_result.scalar_one()

    return TransformationListResponse(
        items=items,
        total=total,
    )


@router.get(
    "/{transformation_id}",
    response_model=TransformationResponse,
)
async def get_transformation(
    transformation_id: UUID,
    db: DatabaseSession,
) -> Transformation:
    """Retrieve a transformation by ID."""

    result = await db.execute(
        select(Transformation).where(
            Transformation.id == transformation_id
        )
    )

    transformation = result.scalar_one_or_none()

    if transformation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transformation not found.",
        )

    return transformation