from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.intelligence.retrieval.context_retriever import (
    RAGLLMIntegrationService,
)
from app.models_ai.gateway import LLMGateway
from app.models_ai.llm.port import LLMPort
from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
)
from app.rag.context import RAGContextAssembler
from app.rag.schemas import (
    RAGContext,
    RAGQuery,
)
from app.rag.service import RAGRetrievalService


MODEL = LLMModelInfo(
    provider="test-provider",
    model_name="test-model",
)


def build_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Explain the knowledge pipeline.",
            ),
        ],
        model=MODEL,
        temperature=0.2,
        max_tokens=500,
        metadata={
            "request_id": "request-001",
        },
    )


def build_context(
    *,
    query: RAGQuery,
    context_text: str,
    chunk_count: int = 2,
) -> RAGContext:
    return RAGContext(
        query=query,
        chunks=[],
        context_text=context_text,
        metadata={
            "chunk_count": chunk_count,
            "source_chunk_count": chunk_count,
        },
    )


def build_gateway() -> tuple[LLMGateway, Mock]:
    port = Mock(spec=LLMPort)

    port.generate = AsyncMock(
        return_value=LLMResponse(
            text="Generated answer.",
            model=MODEL,
        ),
    )

    gateway = LLMGateway(port)

    return gateway, port


def build_service(
    *,
    context: RAGContext,
) -> tuple[
    RAGLLMIntegrationService,
    Mock,
    Mock,
    Mock,
]:
    rag_service = Mock(
        spec=RAGRetrievalService,
    )

    rag_service.retrieve = AsyncMock(
        return_value=[],
    )

    assembler = Mock(
        spec=RAGContextAssembler,
    )

    assembler.assemble.return_value = context

    gateway, port = build_gateway()

    service = RAGLLMIntegrationService(
        rag_retrieval_service=rag_service,
        context_assembler=assembler,
        llm_gateway=gateway,
    )

    return (
        service,
        rag_service,
        assembler,
        port,
    )


@pytest.mark.asyncio
async def test_retrieve_context_runs_rag_pipeline() -> None:
    query = RAGQuery(
        text="How does retrieval work?",
    )

    context = build_context(
        query=query,
        context_text=(
            "Retrieved Knowledge Context\n"
            "[Chunk 1]\n"
            "Retrieval uses embeddings."
        ),
        chunk_count=1,
    )

    (
        service,
        rag_service,
        assembler,
        _,
    ) = build_service(
        context=context,
    )

    result = await service.retrieve_context(
        query,
    )

    rag_service.retrieve.assert_awaited_once_with(
        query,
    )

    assembler.assemble.assert_called_once_with(
        query=query,
        chunks=[],
    )

    assert result is context


@pytest.mark.asyncio
async def test_generate_injects_rag_context_before_original_messages() -> None:
    query = RAGQuery(
        text="How does retrieval work?",
    )

    context = build_context(
        query=query,
        context_text=(
            "Retrieved Knowledge Context\n"
            "[Chunk 1]\n"
            "Embeddings represent semantic meaning."
        ),
        chunk_count=1,
    )

    (
        service,
        _,
        _,
        port,
    ) = build_service(
        context=context,
    )

    request = build_request()

    response = await service.generate(
        query=query,
        request=request,
    )

    assert response.text == "Generated answer."

    port.generate.assert_awaited_once()

    contextual_request = (
        port.generate.await_args.args[0]
    )

    assert len(contextual_request.messages) == 2

    assert (
        contextual_request.messages[0].role
        == "system"
    )

    assert (
        "Embeddings represent semantic meaning."
        in contextual_request.messages[0].content
    )

    assert (
        contextual_request.messages[1].role
        == "user"
    )

    assert (
        contextual_request.messages[1].content
        == "Explain the knowledge pipeline."
    )


@pytest.mark.asyncio
async def test_generate_does_not_mutate_original_request() -> None:
    query = RAGQuery(
        text="Explain RAG.",
    )

    context = build_context(
        query=query,
        context_text=(
            "Retrieved Knowledge Context\n"
            "[Chunk 1]\n"
            "Knowledge retrieval."
        ),
        chunk_count=1,
    )

    (
        service,
        _,
        _,
        _,
    ) = build_service(
        context=context,
    )

    request = build_request()

    original_messages = list(
        request.messages,
    )
    original_metadata = dict(
        request.metadata,
    )

    await service.generate(
        query=query,
        request=request,
    )

    assert request.messages == original_messages
    assert request.metadata == original_metadata


@pytest.mark.asyncio
async def test_generate_preserves_llm_request_configuration() -> None:
    query = RAGQuery(
        text="Explain RAG.",
    )

    context = build_context(
        query=query,
        context_text="Retrieved knowledge.",
        chunk_count=1,
    )

    (
        service,
        _,
        _,
        port,
    ) = build_service(
        context=context,
    )

    request = build_request()

    await service.generate(
        query=query,
        request=request,
    )

    contextual_request = (
        port.generate.await_args.args[0]
    )

    assert contextual_request.model == request.model
    assert (
        contextual_request.temperature
        == request.temperature
    )
    assert (
        contextual_request.max_tokens
        == request.max_tokens
    )


@pytest.mark.asyncio
async def test_generate_attaches_rag_metadata() -> None:
    query = RAGQuery(
        text="Explain RAG.",
    )

    context = build_context(
        query=query,
        context_text="Retrieved knowledge.",
        chunk_count=2,
    )

    (
        service,
        _,
        _,
        port,
    ) = build_service(
        context=context,
    )

    request = build_request()

    await service.generate(
        query=query,
        request=request,
    )

    contextual_request = (
        port.generate.await_args.args[0]
    )

    assert contextual_request.metadata["request_id"] == (
        "request-001"
    )

    assert contextual_request.metadata["rag"] == {
        "enabled": True,
        "chunk_count": 2,
        "source_chunk_count": 2,
        "context_metadata": {
            "chunk_count": 2,
            "source_chunk_count": 2,
        },
    }


def test_context_message_contains_empty_context_notice() -> None:
    query = RAGQuery(
        text="Question with no knowledge.",
    )

    context = build_context(
        query=query,
        context_text="",
        chunk_count=0,
    )

    message = (
        RAGLLMIntegrationService._build_context_message(
            context,
        )
    )

    assert message.role == "system"

    assert (
        "No retrieved knowledge context was found."
        in message.content
    )

    assert message.metadata == {
        "source": "rag",
        "chunk_count": 0,
    }


def test_context_message_preserves_context_text() -> None:
    query = RAGQuery(
        text="Question.",
    )

    context = build_context(
        query=query,
        context_text=(
            "Retrieved Knowledge Context\n"
            "[Chunk 1]\n"
            "Important fact."
        ),
        chunk_count=1,
    )

    message = (
        RAGLLMIntegrationService._build_context_message(
            context,
        )
    )

    assert "Important fact." in message.content
    assert "[Chunk 1]" in message.content


def test_invalid_rag_service_is_rejected() -> None:
    with pytest.raises(TypeError) as exc_info:
        RAGLLMIntegrationService(
            rag_retrieval_service=Mock(),
            context_assembler=RAGContextAssembler(),
            llm_gateway=Mock(),
        )

    assert str(exc_info.value) == (
        "rag_retrieval_service must be a "
        "RAGRetrievalService."
    )


def test_invalid_context_assembler_is_rejected() -> None:
    valid_rag_service = Mock(
        spec=RAGRetrievalService,
    )

    with pytest.raises(TypeError) as exc_info:
        RAGLLMIntegrationService(
            rag_retrieval_service=valid_rag_service,
            context_assembler=Mock(),
            llm_gateway=Mock(
                spec=LLMGateway,
            ),
        )

    assert str(exc_info.value) == (
        "context_assembler must be a "
        "RAGContextAssembler."
    )


def test_invalid_gateway_is_rejected() -> None:
    valid_rag_service = Mock(
        spec=RAGRetrievalService,
    )

    with pytest.raises(TypeError) as exc_info:
        RAGLLMIntegrationService(
            rag_retrieval_service=valid_rag_service,
            context_assembler=RAGContextAssembler(),
            llm_gateway=Mock(),
        )

    assert str(exc_info.value) == (
        "llm_gateway must be an LLMGateway."
    )