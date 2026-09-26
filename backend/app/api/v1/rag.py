from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import RAGRetrievalServiceDependency
from app.rag.context import RAGContextAssembler
from app.rag.schemas import RAGQuery
from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
)

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
)
async def query_rag(
    request: RAGQueryRequest,
    rag_service: RAGRetrievalServiceDependency,
) -> RAGQueryResponse:
    """
    Retrieve relevant knowledge and assemble bounded RAG context.
    """

    try:
        query = RAGQuery.model_validate(
            request.model_dump(),
        )

        chunks = await rag_service.retrieve(
            query,
        )

        context = RAGContextAssembler().assemble(
            query=query,
            chunks=chunks,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return RAGQueryResponse.from_context(
        context=context,
    )