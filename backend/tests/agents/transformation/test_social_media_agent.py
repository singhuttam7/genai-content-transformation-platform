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
from app.agents.transformation.social_media_agent import (
    SocialMediaTransformationAgent,
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


def create_successful_integration(
    output_text: str = "Generated social media post.",
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
) -> SocialMediaTransformationAgent:
    return SocialMediaTransformationAgent(
        integration_service=(
            service
            or create_successful_integration()
        ),
        model=create_model(),
    )


def create_request(
    *,
    input_value: object = "Important source material.",
    platform: str = "LinkedIn",
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
        "platform": platform,
    }

    if configuration:
        config.update(configuration)

    return TransformationRequest(
        transformation_type=TransformationType.SOCIAL_MEDIA,
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


def get_execute_call(
    service: AgentIntegrationService,
):
    assert service.execute.call_count == 1
    return service.execute.call_args


def test_transformation_type() -> None:
    agent = create_agent()

    assert (
        agent.transformation_type
        == TransformationType.SOCIAL_MEDIA
    )


def test_constructor_rejects_invalid_integration_service() -> None:
    with pytest.raises(TypeError):
        SocialMediaTransformationAgent(
            integration_service=object(),  # type: ignore[arg-type]
            model=create_model(),
        )


def test_constructor_rejects_invalid_model() -> None:
    with pytest.raises(TypeError):
        SocialMediaTransformationAgent(
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
async def test_execute_returns_one_social_media_artifact() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert len(result.artifacts) == 1
    assert (
        result.artifacts[0].artifact_type
        == TransformationType.SOCIAL_MEDIA.value
    )


@pytest.mark.asyncio
async def test_generated_content_is_preserved() -> None:
    service = create_successful_integration(
        "Generated LinkedIn content."
    )

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    assert (
        result.artifacts[0].content
        == "Generated LinkedIn content."
    )


@pytest.mark.asyncio
async def test_model_is_passed_to_integration() -> None:
    service = create_successful_integration()

    model = create_model()

    agent = SocialMediaTransformationAgent(
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
async def test_platform_is_forwarded() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            platform="X",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert agent_request.metadata["platform"] == "X"
    assert "X" in agent_request.input


@pytest.mark.asyncio
async def test_task_is_social_media_specific() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            platform="LinkedIn",
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert (
        agent_request.task
        == (
            "Transform the supplied source material into "
            "platform-optimized social media content for LinkedIn."
        )
    )


@pytest.mark.asyncio
async def test_source_content_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            input_value=(
                "Our new AI platform reduced processing time."
            ),
        )
    )

    call = get_execute_call(service)

    agent_request = call.args[0]

    assert (
        "Our new AI platform reduced processing time."
        in agent_request.input
    )


@pytest.mark.asyncio
async def test_objective_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            objective="Promote the platform",
        )
    )

    call = get_execute_call(service)

    assert (
        "Promote the platform"
        in call.args[0].input
    )


@pytest.mark.asyncio
async def test_audience_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            audience="Technology professionals",
        )
    )

    call = get_execute_call(service)

    assert (
        "Technology professionals"
        in call.args[0].input
    )


@pytest.mark.asyncio
async def test_tone_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            tone="Professional",
        )
    )

    call = get_execute_call(service)

    assert (
        "Professional"
        in call.args[0].input
    )


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

    assert "Hindi" in call.args[0].input


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

    assert "Concise" in call.args[0].input


@pytest.mark.asyncio
async def test_style_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            style="Professional",
        )
    )

    call = get_execute_call(service)

    assert "Professional" in call.args[0].input


@pytest.mark.asyncio
async def test_configuration_is_included() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            configuration={
                "include_hashtags": True,
                "max_length": 280,
            },
        )
    )

    call = get_execute_call(service)

    assert "include_hashtags" in call.args[0].input
    assert "max_length" in call.args[0].input


@pytest.mark.asyncio
async def test_request_metadata_is_forwarded() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request(
            metadata={
                "source_id": "source-123",
            },
        )
    )

    call = get_execute_call(service)

    metadata = call.args[0].metadata

    assert metadata["source_id"] == "source-123"
    assert (
        metadata["transformation_type"]
        == "social_media"
    )
    assert metadata["language"] == "English"
    assert metadata["platform"] == "LinkedIn"


@pytest.mark.asyncio
async def test_artifact_metadata_contains_options() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request(
            platform="LinkedIn",
            objective="Launch announcement",
            audience="Developers",
            tone="Professional",
            language="English",
            detail_level="Concise",
            style="Technical",
        )
    )

    metadata = result.artifacts[0].metadata

    assert metadata["platform"] == "LinkedIn"
    assert metadata["objective"] == "Launch announcement"
    assert metadata["audience"] == "Developers"
    assert metadata["tone"] == "Professional"
    assert metadata["language"] == "English"
    assert metadata["detail_level"] == "Concise"
    assert metadata["style"] == "Technical"


@pytest.mark.asyncio
async def test_artifact_provenance_contains_model() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request()
    )

    provenance = result.artifacts[0].provenance

    assert (
        provenance["agent"]
        == "SocialMediaTransformationAgent"
    )
    assert provenance["model_provider"] == "test-provider"
    assert provenance["model_name"] == "test-model"
    assert provenance["rag_enabled"] is True


@pytest.mark.asyncio
async def test_result_metadata_contains_platform() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request(
            platform="X",
        )
    )

    assert result.metadata["platform"] == "X"
    assert result.metadata["rag_enabled"] is True
    assert result.metadata["model_provider"] == "test-provider"
    assert result.metadata["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_title_uses_objective() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request(
            platform="LinkedIn",
            objective="Product launch",
        )
    )

    assert (
        result.artifacts[0].title
        == "LinkedIn Post — Product launch"
    )


@pytest.mark.asyncio
async def test_default_title() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    result = await agent.execute(
        create_request(
            platform="X",
        )
    )

    assert (
        result.artifacts[0].title
        == "Generated X Post"
    )


@pytest.mark.asyncio
async def test_default_platform_is_linkedin() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = TransformationRequest(
        transformation_type=TransformationType.SOCIAL_MEDIA,
        input="Source content",
        configuration={},
    )

    result = await agent.execute(request)

    assert result.metadata["platform"] == "LinkedIn"


@pytest.mark.asyncio
async def test_invalid_platform_type_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = TransformationRequest(
        transformation_type=TransformationType.SOCIAL_MEDIA,
        input="Source content",
        configuration={
            "platform": 123,
        },
    )

    with pytest.raises(
        TypeError,
        match="platform must be a string",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_blank_platform_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = TransformationRequest(
        transformation_type=TransformationType.SOCIAL_MEDIA,
        input="Source content",
        configuration={
            "platform": "   ",
        },
    )

    with pytest.raises(
        ValueError,
        match="platform must not be empty",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_wrong_transformation_type_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source content",
    )

    with pytest.raises(
        ValueError,
        match="only accepts SOCIAL_MEDIA",
    ):
        await agent.execute(request)


@pytest.mark.asyncio
async def test_none_input_is_rejected() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    request = TransformationRequest(
        transformation_type=TransformationType.SOCIAL_MEDIA,
        input=None,
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

    request = TransformationRequest(
        transformation_type=TransformationType.SOCIAL_MEDIA,
        input="   ",
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

    content = call.args[0].input

    assert '"a": 1' in content
    assert '"z": 2' in content
    assert '"value": "important"' in content


@pytest.mark.asyncio
async def test_social_media_requirements_are_present() -> None:
    service = create_successful_integration()

    agent = create_agent(service)

    await agent.execute(
        create_request()
    )

    call = get_execute_call(service)

    content = call.args[0].input

    assert (
        "SOCIAL MEDIA REQUIREMENTS:"
        in content
    )

    assert (
        "Preserve the factual meaning of the source."
        in content
    )

    assert (
        "Do not invent unsupported facts or claims."
        in content
    )

    assert (
        "Optimize the content for the specified platform."
        in content
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
async def test_execution_is_deterministic() -> None:
    service = create_successful_integration(
        "Stable social media content."
    )

    agent = create_agent(service)

    request = create_request(
        input_value={
            "title": "AI Platform",
            "finding": "Processing time decreased",
        },
        platform="LinkedIn",
        objective="Product announcement",
        audience="Technology professionals",
        tone="Professional",
        language="English",
        detail_level="Concise",
        style="Technical",
        configuration={
            "include_hashtags": True,
        },
        metadata={
            "source": "test",
        },
    )

    first = await agent.execute(request)

    service.execute.reset_mock()

    second = await agent.execute(request)

    assert first.model_dump() == second.model_dump()