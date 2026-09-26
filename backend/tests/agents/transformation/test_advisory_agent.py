from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.base import AgentIntegrationService
from app.agents.transformation import (
    AdvisoryTransformationAgent,
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.models_ai.llm import (
    LLMModelInfo,
    LLMResponse,
)


def make_model() -> LLMModelInfo:
    return LLMModelInfo(
        provider="test-provider",
        model_name="test-model",
    )


def make_request(
    *,
    input_value: object = "Important source information.",
    **kwargs: object,
) -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input=input_value,
        **kwargs,
    )


def make_response(
    text: str = "Generated advisory content.",
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model=make_model(),
        usage=None,
        finish_reason="stop",
        metadata={
            "provider": "test-provider",
        },
    )


def make_service(
    response: LLMResponse | None = None,
) -> Mock:
    service = Mock(
        spec=AgentIntegrationService,
    )
    service.execute = AsyncMock(
        return_value=(
            Mock(
                output=response or make_response(),
            )
        ),
    )
    return service


def make_agent(
    service: Mock | None = None,
) -> AdvisoryTransformationAgent:
    return AdvisoryTransformationAgent(
        integration_service=(
            service or make_service()
        ),
        model=make_model(),
    )


def test_agent_exposes_advisory_type() -> None:
    agent = make_agent()

    assert (
        agent.transformation_type
        == TransformationType.ADVISORY
    )


def test_constructor_rejects_invalid_integration_service() -> None:
    with pytest.raises(
        TypeError,
        match="AgentIntegrationService",
    ):
        AdvisoryTransformationAgent(
            integration_service=object(),  # type: ignore[arg-type]
            model=make_model(),
        )


def test_constructor_rejects_invalid_model() -> None:
    with pytest.raises(
        TypeError,
        match="LLMModelInfo",
    ):
        AdvisoryTransformationAgent(
            integration_service=make_service(),
            model=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execute_returns_transformation_result() -> None:
    agent = make_agent()

    result = await agent.execute(
        make_request(),
    )

    assert isinstance(
        result,
        TransformationResult,
    )

    assert (
        result.status
        == TransformationStatus.COMPLETED
    )


@pytest.mark.asyncio
async def test_execute_returns_one_advisory_artifact() -> None:
    agent = make_agent()

    result = await agent.execute(
        make_request(),
    )

    assert len(result.artifacts) == 1

    artifact = result.artifacts[0]

    assert isinstance(
        artifact,
        ArtifactEnvelope,
    )

    assert (
        artifact.artifact_type
        == TransformationType.ADVISORY.value
    )


@pytest.mark.asyncio
async def test_execute_preserves_llm_text_as_artifact_content() -> None:
    service = make_service(
        make_response(
            "Specific advisory output.",
        ),
    )

    agent = make_agent(service)

    result = await agent.execute(
        make_request(),
    )

    assert (
        result.artifacts[0].content
        == "Specific advisory output."
    )


@pytest.mark.asyncio
async def test_execute_passes_model_to_integration_service() -> None:
    service = make_service()

    model = make_model()

    agent = AdvisoryTransformationAgent(
        integration_service=service,
        model=model,
    )

    await agent.execute(
        make_request(),
    )

    service.execute.assert_awaited_once()

    assert (
        service.execute.await_args.kwargs["model"]
        == model
    )


@pytest.mark.asyncio
async def test_execute_enables_rag() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(),
    )

    assert (
        service.execute.await_args.kwargs["use_rag"]
        is True
    )


@pytest.mark.asyncio
async def test_execute_builds_advisory_task() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(),
    )

    agent_request = service.execute.await_args.args[0]

    assert (
        "professional advisory"
        in agent_request.task.lower()
    )


@pytest.mark.asyncio
async def test_execute_includes_source_content() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            input_value="Source document content.",
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert (
        "Source document content."
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_execute_includes_objective() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            objective="Inform stakeholders",
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert (
        "Inform stakeholders"
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_execute_includes_audience() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            audience="Senior administrators",
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert (
        "Senior administrators"
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_execute_includes_tone() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            tone="Formal",
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert "Formal" in agent_request.input


@pytest.mark.asyncio
async def test_execute_includes_language() -> None:
    service = make_service()

    agent = make_agent(service)

    request = make_request(
        language="Hindi",
    )

    await agent.execute(request)

    agent_request = service.execute.await_args.args[0]

    assert "Hindi" in agent_request.input


@pytest.mark.asyncio
async def test_execute_includes_detail_level() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            detail_level="Detailed",
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert "Detailed" in agent_request.input


@pytest.mark.asyncio
async def test_execute_includes_style() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            style="Government advisory",
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert (
        "Government advisory"
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_execute_includes_configuration() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            configuration={
                "include_actions": True,
                "priority": "high",
            },
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert "include_actions" in agent_request.input
    assert "priority" in agent_request.input


@pytest.mark.asyncio
async def test_execute_preserves_request_metadata() -> None:
    service = make_service()

    agent = make_agent(service)

    request = make_request(
        metadata={
            "request_id": "req-001",
            "source_id": "source-001",
        },
    )

    await agent.execute(request)

    agent_request = service.execute.await_args.args[0]

    assert (
        agent_request.metadata["request_id"]
        == "req-001"
    )

    assert (
        agent_request.metadata["source_id"]
        == "source-001"
    )

    assert (
        agent_request.metadata["transformation_type"]
        == "advisory"
    )


@pytest.mark.asyncio
async def test_artifact_contains_transformation_metadata() -> None:
    agent = make_agent()

    result = await agent.execute(
        make_request(
            objective="Notify users",
            audience="Users",
            tone="Professional",
        ),
    )

    metadata = result.artifacts[0].metadata

    assert metadata["transformation_type"] == "advisory"
    assert metadata["objective"] == "Notify users"
    assert metadata["audience"] == "Users"
    assert metadata["tone"] == "Professional"


@pytest.mark.asyncio
async def test_artifact_contains_provenance() -> None:
    agent = make_agent()

    result = await agent.execute(
        make_request(),
    )

    provenance = result.artifacts[0].provenance

    assert (
        provenance["agent"]
        == "AdvisoryTransformationAgent"
    )

    assert provenance["model_provider"] == "test-provider"
    assert provenance["model_name"] == "test-model"
    assert provenance["rag_enabled"] is True


@pytest.mark.asyncio
async def test_result_contains_model_metadata() -> None:
    agent = make_agent()

    result = await agent.execute(
        make_request(),
    )

    assert (
        result.metadata["model_provider"]
        == "test-provider"
    )

    assert (
        result.metadata["model_name"]
        == "test-model"
    )

    assert result.metadata["rag_enabled"] is True


@pytest.mark.asyncio
async def test_llm_failure_is_propagated() -> None:
    service = Mock(
        spec=AgentIntegrationService,
    )

    service.execute = AsyncMock(
        side_effect=RuntimeError(
            "LLM generation failed",
        ),
    )

    agent = make_agent(service)

    with pytest.raises(
        RuntimeError,
        match="LLM generation failed",
    ):
        await agent.execute(
            make_request(),
        )


@pytest.mark.asyncio
async def test_invalid_llm_response_is_rejected() -> None:
    service = Mock(
        spec=AgentIntegrationService,
    )

    service.execute = AsyncMock(
        return_value=Mock(
            output=object(),
        ),
    )

    agent = make_agent(service)

    with pytest.raises(
        TypeError,
        match="invalid LLM response",
    ):
        await agent.execute(
            make_request(),
        )


@pytest.mark.asyncio
async def test_wrong_transformation_type_is_rejected() -> None:
    agent = make_agent()

    request = TransformationRequest(
        transformation_type=(
            TransformationType.SOCIAL_MEDIA
        ),
        input="Source content",
    )

    with pytest.raises(
        ValueError,
        match="only accepts.*ADVISORY",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_none_input_is_rejected() -> None:
    agent = make_agent()

    request = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input=None,
    )

    with pytest.raises(
        ValueError,
        match="must not be None",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_blank_string_input_is_rejected() -> None:
    agent = make_agent()

    request = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="   ",
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_structured_input_is_supported() -> None:
    service = make_service()

    agent = make_agent(service)

    await agent.execute(
        make_request(
            input_value={
                "title": "Important notice",
                "facts": [
                    "Fact one",
                    "Fact two",
                ],
            },
        ),
    )

    agent_request = service.execute.await_args.args[0]

    assert "Important notice" in agent_request.input
    assert "Fact one" in agent_request.input


@pytest.mark.asyncio
async def test_execution_is_deterministic_for_same_request() -> None:
    service = make_service()

    agent = make_agent(service)

    request = make_request(
        objective="Inform users",
        audience="Users",
        tone="Formal",
        language="English",
    )

    await agent.execute(request)
    first_request = service.execute.await_args.args[0]

    await agent.execute(request)
    second_request = service.execute.await_args.args[0]

    assert first_request.task == second_request.task
    assert first_request.input == second_request.input
    assert first_request.metadata == second_request.metadata