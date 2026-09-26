from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.base import (
    AgentExecutionContext,
    AgentRequest,
)


def create_request() -> AgentRequest:
    return AgentRequest(
        task="summarize",
        input="Test document",
    )


def test_execution_context_accepts_request():
    request = create_request()

    context = AgentExecutionContext(
        request=request,
    )

    assert context.request == request
    assert context.rag_context is None
    assert context.metadata == {}


def test_execution_context_accepts_rag_context():
    context = AgentExecutionContext(
        request=create_request(),
        rag_context="Relevant retrieved context.",
    )

    assert context.rag_context == "Relevant retrieved context."


def test_execution_context_accepts_metadata():
    context = AgentExecutionContext(
        request=create_request(),
        metadata={
            "workflow_id": "workflow-1",
            "agent": "summarizer",
        },
    )

    assert context.metadata == {
        "workflow_id": "workflow-1",
        "agent": "summarizer",
    }


def test_execution_context_requires_request():
    with pytest.raises(ValidationError):
        AgentExecutionContext()


def test_execution_context_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AgentExecutionContext(
            request=create_request(),
            unexpected_field="value",
        )


def test_execution_context_rejects_invalid_request():
    with pytest.raises(ValidationError):
        AgentExecutionContext(
            request={
                "task": "",
                "input": "Test",
            },
        )


def test_execution_context_preserves_request_metadata():
    request = AgentRequest(
        task="transform",
        input={"document": "test"},
        metadata={
            "source": "upload",
        },
    )

    context = AgentExecutionContext(
        request=request,
    )

    assert context.request.metadata == {
        "source": "upload",
    }