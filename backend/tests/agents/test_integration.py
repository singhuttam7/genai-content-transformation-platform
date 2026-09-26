from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.base import (
    AgentExecutionContext,
    AgentIntegrationService,
    AgentRAGContextIntegrator,
    AgentRequest,
)
from app.models_ai.gateway import LLMGateway
from app.models_ai.llm import (
    LLMResponse,
    LLMModelInfo,
)


def make_model() -> LLMModelInfo:
    return LLMModelInfo(
        provider="test-provider",
        model_name="test-model",
    )


def make_request() -> AgentRequest:
    return AgentRequest(
        task="Summarize the document.",
        input="Important project information.",
        metadata={
            "request_id": "test-request",
        },
    )


def make_gateway(response: LLMResponse) -> LLMGateway:
    gateway = Mock(spec=LLMGateway)
    gateway.generate = AsyncMock(
        return_value=response,
    )
    return gateway


def make_response() -> LLMResponse:
    return LLMResponse(
        text="Generated summary.",
        model=make_model(),
        usage=None,
        finish_reason="stop",
        metadata={
            "provider": "test-provider",
        },
    )


def make_rag_integrator(
    enriched_context: AgentExecutionContext,
) -> AgentRAGContextIntegrator:
    integrator = Mock(
        spec=AgentRAGContextIntegrator,
    )
    integrator.enrich = AsyncMock(
        return_value=enriched_context,
    )
    return integrator


def test_constructor_rejects_invalid_rag_integrator() -> None:
    with pytest.raises(
        TypeError,
        match="AgentRAGContextIntegrator",
    ):
        AgentIntegrationService(
            rag_integrator=object(),  # type: ignore[arg-type]
            llm_gateway=Mock(spec=LLMGateway),
        )


def test_constructor_rejects_invalid_gateway() -> None:
    with pytest.raises(
        TypeError,
        match="LLMGateway",
    ):
        AgentIntegrationService(
            rag_integrator=Mock(
                spec=AgentRAGContextIntegrator,
            ),
            llm_gateway=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execute_runs_rag_then_llm() -> None:
    request = make_request()

    enriched_context = AgentExecutionContext(
        request=request,
        rag_context="Retrieved knowledge.",
        metadata={
            "rag": {
                "enabled": True,
            },
        },
    )

    rag_integrator = make_rag_integrator(
        enriched_context,
    )

    gateway = make_gateway(
        make_response(),
    )

    service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=gateway,
    )

    result = await service.execute(
        request,
        model=make_model(),
    )

    rag_integrator.enrich.assert_awaited_once()
    gateway.generate.assert_awaited_once()

    assert result.status == "completed"
    assert result.output.text == "Generated summary."
    assert result.metadata["rag_enabled"] is True


@pytest.mark.asyncio
async def test_execute_can_disable_rag() -> None:
    request = make_request()

    rag_integrator = Mock(
        spec=AgentRAGContextIntegrator,
    )
    rag_integrator.enrich = AsyncMock()

    gateway = make_gateway(
        make_response(),
    )

    service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=gateway,
    )

    result = await service.execute(
        request,
        model=make_model(),
        use_rag=False,
    )

    rag_integrator.enrich.assert_not_awaited()
    gateway.generate.assert_awaited_once()

    assert result.status == "completed"
    assert result.metadata["rag_enabled"] is False


@pytest.mark.asyncio
async def test_execute_passes_original_request_to_rag() -> None:
    request = make_request()

    enriched_context = AgentExecutionContext(
        request=request,
        rag_context="Knowledge.",
    )

    rag_integrator = make_rag_integrator(
        enriched_context,
    )

    gateway = make_gateway(
        make_response(),
    )

    service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=gateway,
    )

    await service.execute(
        request,
        model=make_model(),
    )

    passed_context = (
        rag_integrator.enrich.await_args.args[0]
    )

    assert passed_context.request == request


@pytest.mark.asyncio
async def test_execute_maps_enriched_context_to_llm_request() -> None:
    request = make_request()

    enriched_context = AgentExecutionContext(
        request=request,
        rag_context="Retrieved knowledge.",
    )

    rag_integrator = make_rag_integrator(
        enriched_context,
    )

    gateway = make_gateway(
        make_response(),
    )

    service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=gateway,
    )

    await service.execute(
        request,
        model=make_model(),
    )

    llm_request = gateway.generate.await_args.args[0]

    assert llm_request.model.model_name == "test-model"
    assert llm_request.model.provider == "test-provider"

    assert llm_request.messages[0].role == "system"
    assert "Retrieved knowledge." in (
        llm_request.messages[0].content
    )


@pytest.mark.asyncio
async def test_execute_propagates_rag_failure() -> None:
    request = make_request()

    rag_integrator = Mock(
        spec=AgentRAGContextIntegrator,
    )
    rag_integrator.enrich = AsyncMock(
        side_effect=RuntimeError("RAG failed"),
    )

    gateway = make_gateway(
        make_response(),
    )

    service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=gateway,
    )

    with pytest.raises(
        RuntimeError,
        match="RAG failed",
    ):
        await service.execute(
            request,
            model=make_model(),
        )

    gateway.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_propagates_llm_failure() -> None:
    request = make_request()

    enriched_context = AgentExecutionContext(
        request=request,
        rag_context="Knowledge.",
    )

    rag_integrator = make_rag_integrator(
        enriched_context,
    )

    gateway = Mock(spec=LLMGateway)
    gateway.generate = AsyncMock(
        side_effect=RuntimeError("LLM failed"),
    )

    service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=gateway,
    )

    with pytest.raises(
        RuntimeError,
        match="LLM failed",
    ):
        await service.execute(
            request,
            model=make_model(),
        )