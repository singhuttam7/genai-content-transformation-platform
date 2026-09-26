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
from app.agents.transformation.video_agent import (
    VideoTransformationAgent,
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


def create_service(
    output_text: str = "Generated video package.",
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
) -> VideoTransformationAgent:
    return VideoTransformationAgent(
        integration_service=(
            service or create_service()
        ),
        model=create_model(),
    )


def create_request(
    *,
    input_value: object = "Important source material.",
    video_type: str = "explainer",
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
        "video_type": video_type,
    }

    if configuration:
        config.update(configuration)

    return TransformationRequest(
        transformation_type=TransformationType.VIDEO,
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


def test_transformation_type() -> None:
    assert (
        create_agent().transformation_type
        == TransformationType.VIDEO
    )


def test_invalid_integration_service() -> None:
    with pytest.raises(TypeError):
        VideoTransformationAgent(
            integration_service=object(),  # type: ignore[arg-type]
            model=create_model(),
        )


def test_invalid_model() -> None:
    with pytest.raises(TypeError):
        VideoTransformationAgent(
            integration_service=create_integration_service(),
            model=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execute_completed() -> None:
    result = await create_agent().execute(
        create_request()
    )

    assert result.status == TransformationStatus.COMPLETED


@pytest.mark.asyncio
async def test_one_video_artifact() -> None:
    result = await create_agent().execute(
        create_request()
    )

    assert len(result.artifacts) == 1
    assert (
        result.artifacts[0].artifact_type
        == TransformationType.VIDEO.value
    )


@pytest.mark.asyncio
async def test_content_preserved() -> None:
    service = create_service(
        "Complete video production package."
    )

    result = await create_agent(service).execute(
        create_request()
    )

    assert (
        result.artifacts[0].content
        == "Complete video production package."
    )


@pytest.mark.asyncio
async def test_model_forwarded() -> None:
    service = create_service()
    model = create_model()

    agent = VideoTransformationAgent(
        integration_service=service,
        model=model,
    )

    await agent.execute(create_request())

    assert service.execute.call_args.kwargs["model"] is model


@pytest.mark.asyncio
async def test_rag_enabled() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request()
    )

    assert (
        service.execute.call_args.kwargs["use_rag"]
        is True
    )


@pytest.mark.asyncio
async def test_video_type_forwarded() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(video_type="documentary")
    )

    call = service.execute.call_args
    request = call.args[0]

    assert (
        request.metadata["video_type"]
        == "documentary"
    )
    assert "documentary" in request.input


@pytest.mark.asyncio
async def test_video_specific_task() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(video_type="tutorial")
    )

    task = service.execute.call_args.args[0].task

    assert task == (
        "Transform the supplied source material into a "
        "complete, structured video production package "
        "using a tutorial video format."
    )


@pytest.mark.asyncio
async def test_source_is_included() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(
            input_value="The system reduced processing time."
        )
    )

    content = service.execute.call_args.args[0].input

    assert (
        "The system reduced processing time."
        in content
    )


@pytest.mark.asyncio
async def test_optional_fields_are_included() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(
            objective="Explain quarterly results",
            audience="Executives",
            tone="Professional",
            language="English",
            detail_level="Detailed",
            style="Documentary",
        )
    )

    content = service.execute.call_args.args[0].input

    for value in (
        "Explain quarterly results",
        "Executives",
        "Professional",
        "English",
        "Detailed",
        "Documentary",
    ):
        assert value in content


@pytest.mark.asyncio
async def test_configuration_is_included() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(
            configuration={
                "duration_seconds": 120,
                "include_subtitles": True,
            }
        )
    )

    content = service.execute.call_args.args[0].input

    assert "duration_seconds" in content
    assert "120" in content
    assert "include_subtitles" in content


@pytest.mark.asyncio
async def test_metadata_forwarded() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(
            metadata={"source_id": "source-123"}
        )
    )

    metadata = service.execute.call_args.args[0].metadata

    assert metadata["source_id"] == "source-123"
    assert metadata["transformation_type"] == "video"
    assert metadata["language"] == "English"
    assert metadata["video_type"] == "explainer"


@pytest.mark.asyncio
async def test_artifact_metadata() -> None:
    service = create_service()

    result = await create_agent(service).execute(
        create_request(
            video_type="documentary",
            objective="Research briefing",
            audience="Leadership",
        )
    )

    metadata = result.artifacts[0].metadata

    assert metadata["video_type"] == "documentary"
    assert metadata["objective"] == "Research briefing"
    assert metadata["audience"] == "Leadership"


@pytest.mark.asyncio
async def test_provenance() -> None:
    service = create_service()

    result = await create_agent(service).execute(
        create_request()
    )

    provenance = result.artifacts[0].provenance

    assert (
        provenance["agent"]
        == "VideoTransformationAgent"
    )
    assert provenance["model_provider"] == "test-provider"
    assert provenance["model_name"] == "test-model"
    assert provenance["rag_enabled"] is True


@pytest.mark.asyncio
async def test_result_metadata() -> None:
    service = create_service()

    result = await create_agent(service).execute(
        create_request(video_type="tutorial")
    )

    assert result.metadata["video_type"] == "tutorial"
    assert result.metadata["rag_enabled"] is True
    assert result.metadata["model_provider"] == "test-provider"
    assert result.metadata["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_title_uses_objective() -> None:
    service = create_service()

    result = await create_agent(service).execute(
        create_request(
            objective="Explain the platform",
        )
    )

    assert (
        result.artifacts[0].title
        == "Video — Explain the platform"
    )


@pytest.mark.asyncio
async def test_default_title() -> None:
    service = create_service()

    result = await create_agent(service).execute(
        create_request(video_type="documentary")
    )

    assert (
        result.artifacts[0].title
        == "Generated Documentary Video Package"
    )


@pytest.mark.asyncio
async def test_default_video_type() -> None:
    service = create_service()

    request = TransformationRequest(
        transformation_type=TransformationType.VIDEO,
        input="Source",
        configuration={},
    )

    result = await create_agent(service).execute(request)

    assert result.metadata["video_type"] == "explainer"


@pytest.mark.asyncio
async def test_invalid_video_type_rejected() -> None:
    service = create_service()

    request = TransformationRequest(
        transformation_type=TransformationType.VIDEO,
        input="Source",
        configuration={"video_type": 123},
    )

    with pytest.raises(
        TypeError,
        match="Video type must be a string",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_blank_video_type_rejected() -> None:
    service = create_service()

    request = TransformationRequest(
        transformation_type=TransformationType.VIDEO,
        input="Source",
        configuration={"video_type": "   "},
    )

    with pytest.raises(
        ValueError,
        match="Video type must not be empty",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_wrong_transformation_type_rejected() -> None:
    service = create_service()

    request = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source",
    )

    with pytest.raises(
        ValueError,
        match="only accepts VIDEO",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_none_input_rejected() -> None:
    service = create_service()

    request = TransformationRequest(
        transformation_type=TransformationType.VIDEO,
        input=None,
    )

    with pytest.raises(
        ValueError,
        match="input must not be None",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_blank_input_rejected() -> None:
    service = create_service()

    request = TransformationRequest(
        transformation_type=TransformationType.VIDEO,
        input="   ",
    )

    with pytest.raises(
        ValueError,
        match="input must not be empty",
    ):
        await create_agent(service).execute(request)


@pytest.mark.asyncio
async def test_structured_input_serialization() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request(
            input_value={
                "title": "AI Platform",
                "findings": [
                    "Improved processing",
                    "Reduced cost",
                ],
            }
        )
    )

    content = service.execute.call_args.args[0].input

    assert '"title": "AI Platform"' in content
    assert '"Improved processing"' in content
    assert '"Reduced cost"' in content


@pytest.mark.asyncio
async def test_video_requirements_present() -> None:
    service = create_service()

    await create_agent(service).execute(
        create_request()
    )

    content = service.execute.call_args.args[0].input

    assert "VIDEO PRODUCTION REQUIREMENTS:" in content
    assert "Divide the video into logical scenes." in content
    assert "script for every scene" in content
    assert "narration text" in content
    assert "subtitle or caption text" in content
    assert "visual directions" in content
    assert "scene duration" in content


@pytest.mark.asyncio
async def test_llm_failure_propagates() -> None:
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
async def test_invalid_llm_response_rejected() -> None:
    service = create_integration_service()

    service.execute = AsyncMock(
        return_value=SimpleNamespace(
            output="invalid",
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
    service = create_service(
        "Stable video package."
    )

    agent = create_agent(service)

    request = create_request(
        input_value={
            "title": "Platform",
            "findings": [
                "Processing improved",
                "Costs decreased",
            ],
        },
        video_type="explainer",
        objective="Create product video",
        audience="Customers",
        tone="Professional",
        language="English",
        detail_level="Detailed",
        style="Modern",
        configuration={
            "duration_seconds": 120,
            "include_subtitles": True,
        },
        metadata={
            "source": "test",
        },
    )

    first = await agent.execute(request)

    service.execute.reset_mock()

    second = await agent.execute(request)

    assert first.model_dump() == second.model_dump()