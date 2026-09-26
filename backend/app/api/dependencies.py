from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentRegistry
from app.database.session import get_db_session
from app.ingestion.video.vision import VisionService
from app.ingestion.video.vision_dependencies import (
    get_vision_service,
)
from app.orchestration.execution.executor import WorkflowExecutor
from app.rag.service import RAGRetrievalService
from app.storage.dependencies import get_storage_service
from app.storage.service import StorageService


DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db_session),
]


StorageServiceDependency = Annotated[
    StorageService,
    Depends(get_storage_service),
]


VisionServiceDependency = Annotated[
    VisionService,
    Depends(get_vision_service),
]


def get_rag_retrieval_service(
    request: Request,
) -> RAGRetrievalService:
    """
    Resolve the application-configured RAG retrieval service.

    The service is expected to be installed on FastAPI application
    state during application startup.
    """
    service = getattr(
        request.app.state,
        "rag_retrieval_service",
        None,
    )

    if not isinstance(
        service,
        RAGRetrievalService,
    ):
        raise RuntimeError(
            "RAG retrieval service is not configured."
        )

    return service


RAGRetrievalServiceDependency = Annotated[
    RAGRetrievalService,
    Depends(get_rag_retrieval_service),
]


def get_workflow_executor() -> WorkflowExecutor:
    """
    Resolve the application workflow executor.

    The executor uses the domain AgentRegistry and remains responsible
    only for workflow execution. API routes do not implement
    orchestration logic.
    """
    return WorkflowExecutor(
        agent_registry=AgentRegistry(),
    )


WorkflowExecutorDependency = Annotated[
    WorkflowExecutor,
    Depends(get_workflow_executor),
]