from fastapi import APIRouter
from app.api.v1.rag import router as rag_router
from app.api.v1.executions import router as executions_router
from app.api.v1.artifacts import router as artifacts_router
from app.api.v1.health import router as health_router
from app.api.v1.sources import router as sources_router
from app.api.v1.transformations import (
    router as transformations_router,
)


api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(transformations_router)
api_router.include_router(sources_router)
api_router.include_router(rag_router)
api_router.include_router(executions_router)
api_router.include_router(artifacts_router)