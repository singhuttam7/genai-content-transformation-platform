import pytest

from app.agents.base import (
    AgentExecutionContext,
    AgentLLMRequestMapper,
    AgentRequest,
)
from app.models_ai.llm import LLMModelInfo


def create_model() -> LLMModelInfo:
    return LLMModelInfo(
        provider="test-provider",
        model_name="test-model",
    )


def create_context(
    *,
    task: str = "summarize",
    input_value: object = "Test document",
    rag_context: str | None = None,
    request_metadata: dict | None = None,
    context_metadata: dict | None = None,
) -> AgentExecutionContext:
    return AgentExecutionContext(
        request=AgentRequest(
            task=task,
            input=input_value,
            metadata=request_metadata or {},
        ),
        rag_context=rag_context,
        metadata=context_metadata or {},
    )


def test_maps_agent_task_to_user_message():
    context = create_context()

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert request.messages[-1].role == "user"
    assert "Task: summarize" in request.messages[-1].content


def test_maps_agent_input_to_user_message():
    context = create_context(
        input_value="Important document content.",
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert (
        "Input:\nImportant document content."
        in request.messages[-1].content
    )


def test_maps_rag_context_to_system_message():
    context = create_context(
        rag_context="Retrieved knowledge.",
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert request.messages[0].role == "system"
    assert (
        "Retrieved knowledge."
        in request.messages[0].content
    )


def test_omits_empty_rag_context():
    context = create_context(
        rag_context="   ",
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert len(request.messages) == 1
    assert request.messages[0].role == "user"


def test_preserves_request_metadata():
    context = create_context(
        request_metadata={
            "source": "document",
        },
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert request.metadata["source"] == "document"


def test_preserves_context_metadata():
    context = create_context(
        context_metadata={
            "workflow_id": "workflow-1",
        },
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert request.metadata["workflow_id"] == "workflow-1"


def test_context_metadata_overrides_duplicate_request_metadata():
    context = create_context(
        request_metadata={
            "source": "request",
        },
        context_metadata={
            "source": "context",
        },
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert request.metadata["source"] == "context"


def test_adds_agent_task_to_metadata():
    context = create_context(
        task="generate summary",
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert request.metadata["agent_task"] == "generate summary"


def test_serializes_non_string_input():
    context = create_context(
        input_value={
            "title": "Test",
            "value": 42,
        },
    )

    request = AgentLLMRequestMapper.to_request(
        context,
        create_model(),
    )

    assert (
        "Input:\n{'title': 'Test', 'value': 42}"
        in request.messages[-1].content
    )


def test_rejects_invalid_context():
    with pytest.raises(TypeError, match="AgentExecutionContext"):
        AgentLLMRequestMapper.to_request(
            "invalid",
            create_model(),
        )


def test_rejects_invalid_model():
    context = create_context()

    with pytest.raises(TypeError, match="LLMModelInfo"):
        AgentLLMRequestMapper.to_request(
            context,
            "invalid",
        )


def test_rejects_none_input():
    context = create_context(
        input_value=None,
    )

    with pytest.raises(
        ValueError,
        match="Agent input must not be None",
    ):
        AgentLLMRequestMapper.to_request(
            context,
            create_model(),
        )


def test_rejects_empty_string_input():
    context = create_context(
        input_value="   ",
    )

    with pytest.raises(
        ValueError,
        match="Agent input string must not be empty",
    ):
        AgentLLMRequestMapper.to_request(
            context,
            create_model(),
        )


def test_rejects_empty_task():
    context = create_context(
        task="   ",
    )

    with pytest.raises(
        ValueError,
        match="Agent task must not be empty",
    ):
        AgentLLMRequestMapper.to_request(
            context,
            create_model(),
        )


def test_preserves_model():
    model = create_model()

    context = create_context()

    request = AgentLLMRequestMapper.to_request(
        context,
        model,
    )

    assert request.model == model
    assert request.model.provider == "test-provider"
    assert request.model.model_name == "test-model"


def test_mapping_is_deterministic():
    model = create_model()

    context = create_context(
        rag_context="Knowledge",
        request_metadata={
            "source": "test",
        },
        context_metadata={
            "workflow": "workflow-1",
        },
    )

    first = AgentLLMRequestMapper.to_request(
        context,
        model,
    )

    second = AgentLLMRequestMapper.to_request(
        context,
        model,
    )

    assert first == second