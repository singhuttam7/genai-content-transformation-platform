from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.base import (
    AgentIntegrationService,
    AgentRAGContextIntegrator,
)
from app.agents.base.integration import LLMGateway
from app.agents.transformation.contracts import (
    TransformationRequest,
    TransformationStatus,
    TransformationType,
)
from app.agents.transformation.executive_summary_agent import (
    ExecutiveSummaryTransformationAgent,
)
from app.models_ai.llm import LLMModelInfo


def create_model() -> LLMModelInfo:
    return LLMModelInfo(
        provider="test-provider",
        model_name="test-model",
    )


def create_integration_service() -> AgentIntegrationService:
    rag_integrator = Mock(
        spec=AgentRAGContextIntegrator,
    )

    llm_gateway = Mock(
        spec=LLMGateway,
    )

    return AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=llm_gateway,
    )


def create_agent(
    integration_service: AgentIntegrationService | None = None,
) -> ExecutiveSummaryTransformationAgent:
    return ExecutiveSummaryTransformationAgent(
        integration_service=(
            integration_service
            or create_integration_service()
        ),
        model=create_model(),
    )


def create_request(
    *,
    input_value: object = "Important source material.",
    objective: str | None = None,
    audience: str | None = None,
    tone: str | None = None,
    language: str = "English",
    detail_level: str | None = None,
    style: str | None = None,
    configuration: dict | None = None,
    metadata: dict | None = None,
) -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.EXECUTIVE_SUMMARY,
        input=input_value,
        objective=objective,
        audience=audience,
        tone=tone,
        language=language,
        detail_level=detail_level,
        style=style,
        configuration=configuration or {},
        metadata=metadata or {},
    )


def create_successful_integration(
    output_text: str = "Generated executive summary.",
) -> AgentIntegrationService:
    service = create_integration_service()

    service.execute = AsyncMock(
        return_value=SimpleNamespace(
            output=SimpleNamespace(
                text=output_text,
            ),
        ),
    )

    return service


def get_execute_call(
    service: AgentIntegrationService,
):
    assert service.execute.call_count == 1
    return service.execute.call_args


def test_transformation_type() -> None:
    agent = create_agent()

    assert (
        agent.transformation_type
        == TransformationType.EXECUTIVE_SUMMARY
    )


def test_constructor_rejects_invalid_integration_service() -> None:
    with pytest.raises(TypeError):
        ExecutiveSummaryTransformationAgent(
            integration_service=object(),  # type: ignore[arg-type]
            model=create_model(),
        )


def test_constructor_rejects_invalid_model() -> None:
    with pytest.raises(TypeError):
        ExecutiveSummaryTransformationAgent(
            integration_service=create_integration_service(),
            model=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execute_returns_completed_result() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert result.status == TransformationStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_returns_one_executive_summary_artifact() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert len(result.artifacts) == 1

    artifact = result.artifacts[0]

    assert (
        artifact.artifact_type
        == TransformationType.EXECUTIVE_SUMMARY.value
    )


@pytest.mark.asyncio
async def test_generated_content_is_preserved() -> None:
    service = create_successful_integration(
        "Important executive findings."
    )

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert result.artifacts[0].content == (
        "Important executive findings."
    )


@pytest.mark.asyncio
async def test_model_is_passed_to_integration_service() -> None:
    service = create_successful_integration()

    model = create_model()

    agent = ExecutiveSummaryTransformationAgent(
        integration_service=service,
        model=model,
    )

    await agent.execute(
        create_request()
    )

    call = get_execute_call(service)

    assert call.kwargs["model"] is model


@pytest.mark.asyncio
async def test_rag_is_enabled() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request()
    )

    call = get_execute_call(service)

    assert call.kwargs["use_rag"] is True


@pytest.mark.asyncio
async def test_task_is_executive_summary_specific() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request()
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert (
        agent_request.task
        == (
            "Transform the supplied source material into a "
            "concise, decision-oriented executive summary."
        )
    )


@pytest.mark.asyncio
async def test_source_content_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            input_value=(
                "Quarterly revenue increased by 25 percent."
            ),
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert (
        "Quarterly revenue increased by 25 percent."
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_objective_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            objective="Inform senior leadership",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "OBJECTIVE:" in agent_request.input
    assert "Inform senior leadership" in agent_request.input


@pytest.mark.asyncio
async def test_audience_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            audience="Executive leadership",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "AUDIENCE:" in agent_request.input
    assert "Executive leadership" in agent_request.input


@pytest.mark.asyncio
async def test_tone_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            tone="Formal",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "TONE:" in agent_request.input
    assert "Formal" in agent_request.input


@pytest.mark.asyncio
async def test_language_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            language="Hindi",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "LANGUAGE:" in agent_request.input
    assert "Hindi" in agent_request.input


@pytest.mark.asyncio
async def test_detail_level_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            detail_level="Concise",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "DETAIL LEVEL:" in agent_request.input
    assert "Concise" in agent_request.input


@pytest.mark.asyncio
async def test_style_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            style="Board-ready",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "STYLE:" in agent_request.input
    assert "Board-ready" in agent_request.input


@pytest.mark.asyncio
async def test_configuration_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            configuration={
                "max_sections": 5,
                "include_risks": True,
            },
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert "CONFIGURATION:" in agent_request.input
    assert '"include_risks": true' in agent_request.input
    assert '"max_sections": 5' in agent_request.input


@pytest.mark.asyncio
async def test_request_metadata_is_forwarded() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            metadata={
                "source_id": "source-123",
                "workflow_id": "workflow-456",
            },
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert agent_request.metadata["source_id"] == "source-123"
    assert agent_request.metadata["workflow_id"] == "workflow-456"
    assert (
        agent_request.metadata["transformation_type"]
        == "executive_summary"
    )
    assert agent_request.metadata["language"] == "English"


@pytest.mark.asyncio
async def test_artifact_metadata_contains_transformation_options() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request(
            objective="Leadership briefing",
            audience="Board",
            tone="Formal",
            language="English",
            detail_level="Concise",
            style="Executive",
        )
    )

    metadata = result.artifacts[0].metadata

    assert metadata["transformation_type"] == "executive_summary"
    assert metadata["objective"] == "Leadership briefing"
    assert metadata["audience"] == "Board"
    assert metadata["tone"] == "Formal"
    assert metadata["language"] == "English"
    assert metadata["detail_level"] == "Concise"
    assert metadata["style"] == "Executive"


@pytest.mark.asyncio
async def test_artifact_provenance_contains_agent_and_model() -> None:
    service = create_successful_integration()

    model = create_model()

    agent = ExecutiveSummaryTransformationAgent(
        integration_service=service,
        model=model,
    )

    result = await agent.execute(
        create_request()
    )

    provenance = result.artifacts[0].provenance

    assert (
        provenance["agent"]
        == "ExecutiveSummaryTransformationAgent"
    )
    assert provenance["model_provider"] == "test-provider"
    assert provenance["model_name"] == "test-model"
    assert provenance["rag_enabled"] is True


@pytest.mark.asyncio
async def test_result_metadata_contains_model_information() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert (
        result.metadata["transformation_type"]
        == "executive_summary"
    )
    assert result.metadata["rag_enabled"] is True
    assert result.metadata["model_provider"] == "test-provider"
    assert result.metadata["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_title_uses_objective_when_available() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request(
            objective="Review quarterly performance",
        )
    )

    assert (
        result.artifacts[0].title
        == "Executive Summary — Review quarterly performance"
    )


@pytest.mark.asyncio
async def test_default_title_is_used_without_objective() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert (
        result.artifacts[0].title
        == "Generated Executive Summary"
    )


@pytest.mark.asyncio
async def test_llm_failure_is_propagated() -> None:
    service = create_integration_service()

    service.execute = AsyncMock(
        side_effect=RuntimeError("LLM failure")
    )

    agent = create_agent(service)

    with pytest.raises(
        RuntimeError,
        match="LLM failure",
    ):
        await agent.execute(
            create_request()
        )


@pytest.mark.asyncio
async def test_invalid_llm_response_is_rejected() -> None:
    service = create_integration_service()

    service.execute = AsyncMock(
        return_value=SimpleNamespace(
            output="invalid response",
        )
    )

    agent = create_agent(service)

    with pytest.raises(
        TypeError,
        match="invalid LLM response",
    ):
        await agent.execute(
            create_request()
        )


@pytest.mark.asyncio
async def test_wrong_transformation_type_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source material",
    )

    with pytest.raises(
        ValueError,
        match="only accepts EXECUTIVE_SUMMARY",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_none_input_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = create_request(
        input_value=None,
    )

    with pytest.raises(
        ValueError,
        match="input must not be None",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_blank_input_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = create_request(
        input_value="   ",
    )

    with pytest.raises(
        ValueError,
        match="input must not be empty",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_structured_input_is_serialized_deterministically() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            input_value={
                "z": 2,
                "a": 1,
                "nested": {
                    "value": "important",
                },
            },
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert '"a": 1' in agent_request.input
    assert '"z": 2' in agent_request.input
    assert '"value": "important"' in agent_request.input


@pytest.mark.asyncio
async def test_executive_summary_requirements_are_present() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request()
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert (
        "EXECUTIVE SUMMARY REQUIREMENTS:"
        in agent_request.input
    )

    assert (
        "Preserve the factual meaning of the source."
        in agent_request.input
    )

    assert (
        "Focus on the most important information."
        in agent_request.input
    )

    assert (
        "Do not invent unsupported facts"
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_execution_is_deterministic() -> None:
    service = create_successful_integration(
        "Stable executive summary."
    )

    agent = create_agent(service)

    request = create_request(
        input_value={
            "title": "Quarterly Report",
            "findings": [
                "Revenue increased",
                "Costs decreased",
            ],
        },
        objective="Leadership briefing",
        audience="Senior leadership",
        tone="Formal",
        language="English",
        detail_level="Concise",
        style="Board-ready",
        configuration={
            "include_risks": True,
        },
        metadata={
            "source": "test",
        },
    )

    first = await agent.execute(request)

    service.execute.reset_mock()

    second = await agent.execute(request)

    assert first.model_dump() == second.model_dump()