from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.base import (
    AgentExecutionContext,
    AgentRAGContextIntegrator,
    AgentRequest,
)
from app.intelligence.retrieval.context_retriever import (
    RAGLLMIntegrationService,
)
from app.rag.schemas import (
    RAGContext,
    RAGQuery,
)


def make_context(
    *,
    task: str = "Summarize the document.",
    input_value: object = "Important project information.",
    metadata: dict | None = None,
) -> AgentExecutionContext:
    return AgentExecutionContext(
        request=AgentRequest(
            task=task,
            input=input_value,
            metadata={
                "request_id": "test-request",
            },
        ),
        metadata=metadata or {},
    )


def make_rag_service(
    context: RAGContext,
) -> RAGLLMIntegrationService:
    service = RAGLLMIntegrationService.__new__(
        RAGLLMIntegrationService,
    )
    service.retrieve_context = AsyncMock(
        return_value=context,
    )
    return service


def make_rag_context(
    *,
    query: RAGQuery | None = None,
    context_text: str = (
        "Retrieved Knowledge Context\n"
        "[Chunk 1]\n"
        "Important retrieved fact."
    ),
    chunk_count: int = 0,
) -> RAGContext:
    if query is None:
        query = RAGQuery(
            text="Test retrieval query.",
        )

    return RAGContext(
        query=query,
        chunks=[],
        context_text=context_text,
        metadata={
            "chunk_count": chunk_count,
            "source_chunk_count": chunk_count,
            "max_chunks": 5,
            "max_context_characters": 4000,
            "truncated_chunk_count": 0,
            "chunk_ids": [],
            "similarities": [],
            "has_provenance": False,
        },
    )


def test_build_query_uses_agent_task_and_input() -> None:
    context = make_context(
        task="Explain the architecture.",
        input_value="Describe the RAG pipeline.",
    )

    query = AgentRAGContextIntegrator.build_query(
        context,
    )

    assert isinstance(query, RAGQuery)
    assert query.text == (
        "Task: Explain the architecture.\n\n"
        "Input:\nDescribe the RAG pipeline."
    )


def test_build_query_accepts_non_string_input() -> None:
    context = make_context(
        task="Analyze the data.",
        input_value={"records": 10},
    )

    query = AgentRAGContextIntegrator.build_query(
        context,
    )

    assert "Task: Analyze the data." in query.text
    assert "Input:" in query.text
    assert "{'records': 10}" in query.text


def test_build_query_rejects_invalid_context() -> None:
    with pytest.raises(TypeError):
        AgentRAGContextIntegrator.build_query(
            object(),  # type: ignore[arg-type]
        )


def test_build_query_rejects_empty_string_input() -> None:
    context = make_context(
        input_value="   ",
    )

    with pytest.raises(ValueError, match="must not be empty"):
        AgentRAGContextIntegrator.build_query(
            context,
        )


def test_build_query_rejects_none_input() -> None:
    context = make_context(
        input_value=None,
    )

    with pytest.raises(ValueError, match="must not be None"):
        AgentRAGContextIntegrator.build_query(
            context,
        )


def test_constructor_requires_rag_service() -> None:
    with pytest.raises(
        TypeError,
        match="RAGLLMIntegrationService",
    ):
        AgentRAGContextIntegrator(
            rag_service=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_enrich_calls_existing_rag_pipeline() -> None:
    rag_context = make_rag_context(
        context_text="Retrieved project knowledge.",
    )

    service = make_rag_service(
        rag_context,
    )

    integrator = AgentRAGContextIntegrator(
        rag_service=service,
    )

    context = make_context(
        task="Explain the project.",
        input_value="Architecture details.",
    )

    enriched = await integrator.enrich(
        context,
    )

    service.retrieve_context.assert_awaited_once()

    query = service.retrieve_context.await_args.args[0]

    assert isinstance(query, RAGQuery)
    assert query.text == (
        "Task: Explain the project.\n\n"
        "Input:\nArchitecture details."
    )

    assert enriched.rag_context == (
        "Retrieved project knowledge."
    )


@pytest.mark.asyncio
async def test_enrich_preserves_original_request() -> None:
    rag_context = make_rag_context()

    service = make_rag_service(
        rag_context,
    )

    integrator = AgentRAGContextIntegrator(
        rag_service=service,
    )

    context = make_context(
        metadata={
            "existing": True,
        },
    )

    enriched = await integrator.enrich(
        context,
    )

    assert enriched.request == context.request
    assert enriched.request is not context.request
    assert context.rag_context is None
    assert context.metadata == {
        "existing": True,
    }


@pytest.mark.asyncio
async def test_enrich_attaches_rag_metadata() -> None:
    rag_context = make_rag_context(
        context_text="Retrieved knowledge.",
    )

    service = make_rag_service(
        rag_context,
    )

    integrator = AgentRAGContextIntegrator(
        rag_service=service,
    )

    context = make_context(
        metadata={
            "existing": True,
        },
    )

    enriched = await integrator.enrich(
        context,
    )

    assert enriched.metadata["existing"] is True
    assert enriched.metadata["rag"] == {
        "enabled": True,
        "chunk_count": 0,
        "source_chunk_count": 0,
        "context_metadata": rag_context.metadata,
    }


@pytest.mark.asyncio
async def test_enrich_does_not_mutate_metadata() -> None:
    rag_context = make_rag_context()

    service = make_rag_service(
        rag_context,
    )

    integrator = AgentRAGContextIntegrator(
        rag_service=service,
    )

    original_metadata = {
        "source": "agent",
        "nested": {
            "value": 10,
        },
    }

    context = make_context(
        metadata=original_metadata,
    )

    enriched = await integrator.enrich(
        context,
    )

    enriched.metadata["nested"]["value"] = 999

    assert context.metadata == original_metadata


def test_enrich_context_requires_rag_context() -> None:
    context = make_context()

    with pytest.raises(
        TypeError,
        match="RAGContext",
    ):
        AgentRAGContextIntegrator._enrich_context(
            context=context,
            rag_context=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_enrich_propagates_rag_failure() -> None:
    service = RAGLLMIntegrationService.__new__(
        RAGLLMIntegrationService,
    )

    service.retrieve_context = AsyncMock(
        side_effect=RuntimeError("retrieval failed"),
    )

    integrator = AgentRAGContextIntegrator(
        rag_service=service,
    )

    with pytest.raises(
        RuntimeError,
        match="retrieval failed",
    ):
        await integrator.enrich(
            make_context(),
        )