from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.services.development_workflow import (
    get_or_create_development_workflow,
)

router = APIRouter(
    prefix="/development/workflow",
    tags=["development"],
)


@router.post("")
async def create_development_workflow(
    project_id: UUID,
    transformation_type: str = "executive_summary",
    session: AsyncSession = Depends(get_db_session),
):
    try:
        workflow = await get_or_create_development_workflow(
            session=session,
            project_id=project_id,
            transformation_type=transformation_type,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "id": str(workflow.id),
        "project_id": str(workflow.project_id),
        "name": workflow.name,
        "description": workflow.description,
        "version": workflow.version,
        "definition": workflow.definition,
        "is_active": workflow.is_active,
    }