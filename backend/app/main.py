from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description=(
        "GenAI platform for automated transformation of "
        "multimodal source content into communication artifacts."
    ),
    version=settings.app_version,
)

app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }