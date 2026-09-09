from fastapi import APIRouter

from app.api.dependencies import DatabaseSession

from sqlalchemy import text


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
async def health_check(
    db: DatabaseSession,
) -> dict[str, str]:
    """Check application and database health."""

    await db.execute(text("SELECT 1"))

    return {
        "status": "healthy",
        "service": "genai-content-transformation-platform",
        "database": "healthy",
    }