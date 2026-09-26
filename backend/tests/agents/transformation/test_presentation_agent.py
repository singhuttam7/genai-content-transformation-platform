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
from app.agents.transformation.presentation_agent import (
    PresentationTransformationAgent,
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
    output_text: str = "Generated presentation content.",
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
) -> PresentationTransformationAgent:
    return PresentationTransformationAgent(
        integration_service=(
            service or create_successful_integration()
        ),
        model=create_model(),
    )


def create_request(
    *,
    input_value: object = "Important source material.",
    presentation_type: str = "professional",
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
        "presentation_type": presentation_type,
    }

    if configuration:
        config.update(configuration)

    return TransformationRequest(
        transformation_type=TransformationType.PRESENTATION,
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
        == TransformationType.PRESENTATION
    )


def test_constructor_rejects_invalid_integration_service() -> None:
    with pytest.raises(TypeError):
        PresentationTransformationAgent(
            integration_service=object(),  # type: ignore[arg-type]
            model=create_model(),
        )


def test_constructor_rejects_invalid_model() -> None:
    with pytest.raises(TypeError):
        PresentationTransformationAgent(
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
async def test_returns_one_presentation_artifact() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request()
    )

    assert len(result.artifacts) == 1
    assert (
        result.artifacts[0].artifact_type
        == TransformationType.PRESENTATION.value
    )


@pytest.mark.asyncio
async def test_generated_content_is_preserved() -> None:
    service = create_successful_integration(
        "Complete slide package."
    )

    result = await create_agent(service).execute(
        create_request()
    )

    assert (
        result.artifacts[0].content
        == "Complete slide package."
    )


@pytest.mark.asyncio
async def test_model_is_passed_to_integration() -> None:
    service = create_successful_integration()
    model = create_model()

    agent = PresentationTransformationAgent(
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
async def test_presentation_type_is_forwarded() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            presentation_type="executive",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert (
        agent_request.metadata["presentation_type"]
        == "executive"
    )
    assert "executive" in agent_request.input


@pytest.mark.asyncio
async def test_task_is_presentation_specific() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            presentation_type="technical",
        )
    )

    call = get_execute_call(service)

    assert (
        call.args[0].task
        == (
            "Transform the supplied source material into a "
            "complete, structured presentation package using "
            "a technical presentation style."
        )
    )


@pytest.mark.asyncio
async def test_source_content_is_included() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            input_value=(
                "The platform reduced processing time by 40 percent."
            ),
        )
    )

    content = get_execute_call(service).args[0].input

    assert (
        "The platform reduced processing time by 40 percent."
        in content
    )


@pytest.mark.asyncio
async def test_optional_fields_are_included() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            objective="Present quarterly results",
            audience="Senior leadership",
            tone="Professional",
            language="English",
            detail_level="Detailed",
            style="Executive",
        )
    )

    content = get_execute_call(service).args[0].input

    for value in (
        "Present quarterly results",
        "Senior leadership",
        "Professional",
        "English",
        "Detailed",
        "Executive",
    ):
        assert value in content


@pytest.mark.asyncio
async def test_configuration_is_included() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request(
            configuration={
                "slide_count": 10,
                "include_speaker_notes": True,
            }
        )
    )

    content = get_execute_call(service).args[0].input

    assert "slide_count" in content
    assert "10" in content
    assert "include_speaker_notes" in content


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
    assert metadata["transformation_type"] == "presentation"
    assert metadata["language"] == "English"
    assert metadata["presentation_type"] == "professional"


@pytest.mark.asyncio
async def test_artifact_metadata_is_correct() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            presentation_type="executive",
            objective="Quarterly briefing",
            audience="Board",
        )
    )

    metadata = result.artifacts[0].metadata

    assert metadata["presentation_type"] == "executive"
    assert metadata["objective"] == "Quarterly briefing"
    assert metadata["audience"] == "Board"


@pytest.mark.asyncio
async def test_provenance_is_correct() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request()
    )

    provenance = result.artifacts[0].provenance

    assert (
        provenance["agent"]
        == "PresentationTransformationAgent"
    )
    assert provenance["model_provider"] == "test-provider"
    assert provenance["model_name"] == "test-model"
    assert provenance["rag_enabled"] is True


@pytest.mark.asyncio
async def test_result_metadata_is_correct() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            presentation_type="technical",
        )
    )

    assert (
        result.metadata["presentation_type"]
        == "technical"
    )
    assert result.metadata["rag_enabled"] is True
    assert result.metadata["model_provider"] == "test-provider"
    assert result.metadata["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_title_uses_objective() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            objective="Quarterly performance",
        )
    )

    assert (
        result.artifacts[0].title
        == "Presentation — Quarterly performance"
    )


@pytest.mark.asyncio
async def test_default_title_uses_presentation_type() -> None:
    service = create_successful_integration()

    result = await create_agent(service).execute(
        create_request(
            presentation_type="technical",
        )
    )

    assert (
        result.artifacts[0].title
        == "Generated Technical Presentation"
    )


@pytest.mark.asyncio
async def test_default_presentation_type_is_professional() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.PRESENTATION,
        input="Source",
        configuration={},
    )

    result = await create_agent(service).execute(request)

    assert (
        result.metadata["presentation_type"]
        == "professional"
    )


@pytest.mark.asyncio
async def test_invalid_presentation_type_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.PRESENTATION,
        input="Source",
        configuration={
            "presentation_type": 123,
        },
    )

    with pytest.raises(
        TypeError,
        match="Presentation type must be a string",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_blank_presentation_type_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.PRESENTATION,
        input="Source",
        configuration={
            "presentation_type": "   ",
        },
    )

    with pytest.raises(
        ValueError,
        match="Presentation type must not be empty",
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
        match="only accepts PRESENTATION",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_none_input_is_rejected() -> None:
    service = create_successful_integration()

    request = TransformationRequest(
        transformation_type=TransformationType.PRESENTATION,
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
        transformation_type=TransformationType.PRESENTATION,
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
async def test_presentation_requirements_are_present() -> None:
    service = create_successful_integration()

    await create_agent(service).execute(
        create_request()
    )

    content = get_execute_call(service).args[0].input

    assert "PRESENTATION REQUIREMENTS:" in content
    assert (
        "Preserve the factual meaning of the source."
        in content
    )
    assert (
        "Do not invent unsupported facts"
        in content
    )
    assert (
        "speaker notes"
        in content
    )
    assert (
        "charts, diagrams, images"
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
        "Stable presentation package."
    )

    agent = create_agent(service)

    request = create_request(
        input_value={
            "title": "AI Platform",
            "findings": [
                "Processing improved",
                "Costs decreased",
            ],
        },
        presentation_type="executive",
        objective="Leadership briefing",
        audience="Senior leadership",
        tone="Professional",
        language="English",
        detail_level="Detailed",
        style="Board-ready",
        configuration={
            "slide_count": 8,
            "include_speaker_notes": True,
        },
        metadata={
            "source": "test",
        },
    )

    first = await agent.execute(request)

    service.execute.reset_mock()

    second = await agent.execute(request)

    assert first.model_dump() == second.model_dump()