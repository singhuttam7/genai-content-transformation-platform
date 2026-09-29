from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.services.development_workspace import (
    get_or_create_development_workspace,
)


router = APIRouter(
    prefix="/development/workspace",
    tags=["development"],
)


@router.get("")
async def get_development_workspace(
    session: AsyncSession = Depends(get_db_session),
):
    user, project = await get_or_create_development_workspace(
        session,
    )

    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
        "project": {
            "id": str(project.id),
            "name": project.name,
            "description": project.description,
        },
    }