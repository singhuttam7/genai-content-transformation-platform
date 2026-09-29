from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentRegistry
from app.artifacts.service import ArtifactPersistenceService
from app.database.session import get_db_session
from app.ingestion.video.vision import VisionService
from app.ingestion.video.vision_dependencies import (
    get_vision_service,
)
from app.orchestration.execution.executor import WorkflowExecutor
from app.rag.service import RAGRetrievalService
from app.services.agent_runtime import create_agent_registry
from app.services.llm_runtime import (
    create_llm_gateway,
    get_llm_model,
)
from app.services.rag_runtime import create_rag_retrieval_service
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


def get_artifact_persistence_service(
    db: DatabaseSession,
) -> ArtifactPersistenceService:
    return ArtifactPersistenceService(session=db)


ArtifactPersistenceServiceDependency = Annotated[
    ArtifactPersistenceService,
    Depends(get_artifact_persistence_service),
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


async def get_workflow_executor(
    db: DatabaseSession,
) -> WorkflowExecutor:
    """
    Resolve a workflow executor with the application's real
    transformation-agent runtime.

    The registry is constructed per request so the RAG retrieval
    service can use the request-scoped database session.
    """
    rag_retrieval_service = create_rag_retrieval_service(
        db,
    )

    llm_gateway = create_llm_gateway()
    model = get_llm_model()

    agent_registry = create_agent_registry(
        rag_retrieval_service=rag_retrieval_service,
        llm_gateway=llm_gateway,
        model=model,
    )

    if not isinstance(
        agent_registry,
        AgentRegistry,
    ):
        raise RuntimeError(
            "Agent registry was not configured correctly."
        )

    return WorkflowExecutor(
        agent_registry=agent_registry,
    )


WorkflowExecutorDependency = Annotated[
    WorkflowExecutor,
    Depends(get_workflow_executor),
]