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
from app.agents.transformation.infographic_agent import (
    InfographicTransformationAgent,
)
from app.models_ai.llm import LLMModelInfo


def create_model() -> LLMModelInfo:
    return LLMModelInfo(
        provider="test-provider",
        model_name="test-model",
    )


def create_integration_service() -> AgentIntegrationService:
    return AgentIntegrationService(
        rag_integrator=Mock(
            spec=AgentRAGContextIntegrator,
        ),
        llm_gateway=Mock(
            spec=LLMGateway,
        ),
    )


def create_successful_integration(
    output_text: str = "Generated infographic content.",
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


def create_agent(
    service: AgentIntegrationService | None = None,
) -> InfographicTransformationAgent:
    return InfographicTransformationAgent(
        integration_service=(
            service or create_successful_integration()
        ),
        model=create_model(),
    )


def create_request(
    *,
    input_value: object = "Important source material.",
    format_type: str = "standard",
    objective: str | None = None,
    audience: str | None = None,
    tone: str | None = None,
    language: str = "English",
    detail_level: str | None = None,
    style: str | None = None,
    configuration: dict | None = None,
    metadata: dict | None = None,
) -> TransformationRequest:
    config = {
        "format": format_type,
    }

    if configuration:
        config.update(configuration)

    return TransformationRequest(
        transformation_type=TransformationType.INFOGRAPHIC,
        input=input_value,
        objective=objective,
        audience=audience,
        tone=tone,
        language=language,
        detail_level=detail_level,
        style=style,
        configuration=config,
        metadata=metadata or {},
    )


def get_execute_call(service: AgentIntegrationService):
    assert service.execute.call_count == 1
    return service.execute.call_args


def test_transformation_type() -> None:
    agent = create_agent()

    assert (
        agent.transformation_type
        == TransformationType.INFOGRAPHIC
    )


def test_constructor_rejects_invalid_integration_service() -> None:
    with pytest.raises(TypeError):
        InfographicTransformationAgent(
            integration_service=object(),  # type: ignore[arg-type]
            model=create_model(),
        )


def test_constructor_rejects_invalid_model() -> None:
    with pytest.raises(TypeError):
        InfographicTransformationAgent(
            integration_service=create_integration_service(),
            model=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execute_returns_completed_result() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request()
    )

    assert result.status == TransformationStatus.COMPLETED


@pytest.mark.asyncio
async def test_returns_one_infographic_artifact() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request()
    )

    assert len(result.artifacts) == 1
    assert (
        result.artifacts[0].artifact_type
        == TransformationType.INFOGRAPHIC.value
    )


@pytest.mark.asyncio
async def test_generated_content_is_preserved() -> None:
    service = create_successful_integration(
        "Visual infographic structure."
    )

    result = await create_agent(service).execute(
        create_request()
    )

    assert (
        result.artifacts[0].content
        == "Visual infographic structure."
    )


@pytest.mark.asyncio
async def test_model_is_passed_to_integration() -> None:
    service = create_successful_integration()
    model = create_model()

    agent = InfographicTransformationAgent(
        integration_service=service,
        model=model,
    )

    await agent.execute(create_request())

    call = get_execute_call(service)

    assert call.kwargs["model"] is model


@pytest.mark.asyncio
async def test_rag_is_enabled() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request()
    )

    call = get_execute_call(service)

    assert call.kwargs["use_rag"] is True


@pytest.mark.asyncio
async def test_format_is_forwarded() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            format_type="timeline",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert agent_request.metadata["format"] == "timeline"
    assert "timeline" in agent_request.input


@pytest.mark.asyncio
async def test_task_is_infographic_specific() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            format_type="comparison",
        )
    )

    call = get_execute_call(service)

    assert (
        call.args[0].task
        == (
            "Transform the supplied source material into a "
            "structured, visually coherent infographic content "
            "package using the comparison format."
        )
    )


@pytest.mark.asyncio
async def test_source_content_is_included() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            input_value="Revenue increased by 30 percent.",
        )
    )

    call = get_execute_call(service)

    assert (
        "Revenue increased by 30 percent."
        in call.args[0].input
    )


@pytest.mark.asyncio
async def test_optional_fields_are_included() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            objective="Explain growth",
            audience="Executives",
            tone="Professional",
            language="English",
            detail_level="Concise",
            style="Modern",
        )
    )

    content = get_execute_call(service).args[0].input

    for value in (
        "Explain growth",
        "Executives",
        "Professional",
        "English",
        "Concise",
        "Modern",
    ):
        assert value in content


@pytest.mark.asyncio
async def test_configuration_is_included() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            configuration={
                "orientation": "portrait",
                "include_charts": True,
            }
        )
    )

    content = get_execute_call(service).args[0].input

    assert "orientation" in content
    assert "portrait" in content
    assert "include_charts" in content


@pytest.mark.asyncio
async def test_request_metadata_is_forwarded() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            metadata={
                "source_id": "source-123",
            }
        )
    )

    metadata = get_execute_call(service).args[0].metadata

    assert metadata["source_id"] == "source-123"
    assert metadata["transformation_type"] == "infographic"
    assert metadata["language"] == "English"
    assert metadata["format"] == "standard"


@pytest.mark.asyncio
async def test_artifact_metadata_is_correct() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            format_type="timeline",
            objective="Show milestones",
            audience="Leadership",
        )
    )

    metadata = result.artifacts[0].metadata

    assert metadata["format"] == "timeline"
    assert metadata["objective"] == "Show milestones"
    assert metadata["audience"] == "Leadership"


@pytest.mark.asyncio
async def test_provenance_is_correct() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request()
    )

    provenance = result.artifacts[0].provenance

    assert (
        provenance["agent"]
        == "InfographicTransformationAgent"
    )
    assert provenance["model_provider"] == "test-provider"
    assert provenance["model_name"] == "test-model"
    assert provenance["rag_enabled"] is True


@pytest.mark.asyncio
async def test_result_metadata_is_correct() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            format_type="process",
        )
    )

    assert result.metadata["format"] == "process"
    assert result.metadata["rag_enabled"] is True
    assert result.metadata["model_provider"] == "test-provider"
    assert result.metadata["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_title_uses_objective() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            objective="Explain the process",
        )
    )

    assert (
        result.artifacts[0].title
        == "Infographic — Explain the process"
    )


@pytest.mark.asyncio
async def test_default_title_uses_format() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            format_type="timeline",
        )
    )

    assert (
        result.artifacts[0].title
        == "Generated Timeline Infographic"
    )


@pytest.mark.asyncio
async def test_default_format_is_standard() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.INFOGRAPHIC,
        input="Source",
        configuration={},
    )

    result = await create_agent(service).execute(request)

    assert result.metadata["format"] == "standard"


@pytest.mark.asyncio
async def test_invalid_format_type_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.INFOGRAPHIC,
        input="Source",
        configuration={
            "format": 123,
        },
    )

    with pytest.raises(
        TypeError,
        match="format must be a string",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_blank_format_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.INFOGRAPHIC,
        input="Source",
        configuration={
            "format": "   ",
        },
    )

    with pytest.raises(
        ValueError,
        match="format must not be empty",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_wrong_transformation_type_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source",
    )

    with pytest.raises(
        ValueError,
        match="only accepts INFOGRAPHIC",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_none_input_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.INFOGRAPHIC,
        input=None,
    )

    with pytest.raises(
        ValueError,
        match="input must not be None",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_blank_input_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.INFOGRAPHIC,
        input="   ",
    )

    with pytest.raises(
        ValueError,
        match="input must not be empty",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_structured_input_is_deterministic() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            input_value={
                "z": 2,
                "a": 1,
                "nested": {
                    "value": "important",
                },
            }
        )
    )

    content = get_execute_call(service).args[0].input

    assert '"a": 1' in content
    assert '"z": 2' in content
    assert '"value": "important"' in content


@pytest.mark.asyncio
async def test_infographic_requirements_are_present() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request()
    )

    content = get_execute_call(service).args[0].input

    assert "INFOGRAPHIC REQUIREMENTS:" in content
    assert (
        "Preserve the factual meaning of the source."
        in content
    )
    assert (
        "Do not invent unsupported facts"
        in content
    )
    assert (
        "charts, icons, diagrams"
        in content
    )


@pytest.mark.asyncio
async def test_llm_failure_is_propagated() -> None:
    service = create_integration_service()

    service.execute = AsyncMock(
        side_effect=RuntimeError("LLM failure")
    )

    with pytest.raises(
        RuntimeError,
        match="LLM failure",
    ):
        await create_agent(service).execute(
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

    with pytest.raises(
        TypeError,
        match="invalid LLM response",
    ):
        await create_agent(service).execute(
            create_request()
        )


@pytest.mark.asyncio
async def test_execution_is_deterministic() -> None:
    service = create_successful_integration(
        "Stable infographic package."
    )

    agent = create_agent(service)

    request = create_request(
        input_value={
            "title": "AI Platform",
            "metrics": {
                "speed": "30 percent faster",
            },
        },
        format_type="comparison",
        objective="Explain platform improvements",
        audience="Technology leaders",
        tone="Professional",
        language="English",
        detail_level="Concise",
        style="Modern",
        configuration={
            "include_charts": True,
        },
        metadata={
            "source": "test",
        },
    )

    first = await agent.execute(request)

    service.execute.reset_mock()

    second = await agent.execute(request)

    assert first.model_dump() == second.model_dump()