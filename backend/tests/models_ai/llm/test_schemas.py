from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


def build_model_info() -> LLMModelInfo:
    return LLMModelInfo(
        provider="groq",
        model_name="openai/gpt-oss-120b",
        context_window=128000,
        supports_streaming=True,
        supports_structured_output=True,
        metadata={
            "runtime": "cloud",
        },
    )


class TestLLMModelInfo:
    def test_valid_model_info(self) -> None:
        model = build_model_info()

        assert model.provider == "groq"
        assert model.model_name == "openai/gpt-oss-120b"
        assert model.context_window == 128000
        assert model.supports_streaming is True
        assert model.supports_structured_output is True
        assert model.metadata == {
            "runtime": "cloud",
        }

    def test_optional_capabilities_default(self) -> None:
        model = LLMModelInfo(
            provider="test-provider",
            model_name="test-model",
        )

        assert model.context_window is None
        assert model.supports_streaming is False
        assert model.supports_structured_output is False
        assert model.metadata == {}

    @pytest.mark.parametrize(
        "field",
        ["provider", "model_name"],
    )
    def test_blank_identity_rejected(
        self,
        field: str,
    ) -> None:
        values = {
            "provider": "test-provider",
            "model_name": "test-model",
        }
        values[field] = "   "

        with pytest.raises(ValidationError):
            LLMModelInfo(**values)

    def test_context_window_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LLMModelInfo(
                provider="test-provider",
                model_name="test-model",
                context_window=0,
            )

    def test_model_info_is_immutable(self) -> None:
        model = build_model_info()

        with pytest.raises(ValidationError):
            model.provider = "another-provider"


class TestLLMMessage:
    def test_valid_message(self) -> None:
        message = LLMMessage(
            role="user",
            content="Explain the architecture.",
        )

        assert message.role == "user"
        assert message.content == (
            "Explain the architecture."
        )
        assert message.metadata == {}

    @pytest.mark.parametrize(
        "role",
        [
            "system",
            "user",
            "assistant",
        ],
    )
    def test_supported_roles(
        self,
        role: str,
    ) -> None:
        message = LLMMessage(
            role=role,
            content="Test message.",
        )

        assert message.role == role

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMMessage(
                role="tool",
                content="Tool output.",
            )

    def test_blank_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMMessage(
                role="user",
                content="   ",
            )

    def test_message_is_immutable(self) -> None:
        message = LLMMessage(
            role="user",
            content="Original content.",
        )

        with pytest.raises(ValidationError):
            message.content = "Changed content."


class TestLLMRequest:
    def test_valid_request(self) -> None:
        model = build_model_info()

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                LLMMessage(
                    role="user",
                    content="Explain RAG.",
                ),
            ],
            model=model,
        )

        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"
        assert request.model == model
        assert request.temperature == 0.0
        assert request.max_tokens is None
        assert request.metadata == {}

    def test_custom_generation_options(self) -> None:
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="Generate a summary.",
                ),
            ],
            model=build_model_info(),
            temperature=0.7,
            max_tokens=500,
            metadata={
                "request_id": str(uuid4()),
            },
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 500
        assert "request_id" in request.metadata

    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[],
                model=build_model_info(),
            )

    @pytest.mark.parametrize(
        "temperature",
        [-0.01, 2.01],
    )
    def test_temperature_bounds(
        self,
        temperature: float,
    ) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="user",
                        content="Test.",
                    ),
                ],
                model=build_model_info(),
                temperature=temperature,
            )

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="user",
                        content="Test.",
                    ),
                ],
                model=build_model_info(),
                max_tokens=0,
            )

    def test_request_is_immutable(self) -> None:
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content="Test.",
                ),
            ],
            model=build_model_info(),
        )

        with pytest.raises(ValidationError):
            request.temperature = 1.0


class TestLLMUsage:
    def test_valid_usage(self) -> None:
        usage = LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_fields_are_optional(self) -> None:
        usage = LLMUsage()

        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.total_tokens is None

    def test_negative_usage_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMUsage(
                prompt_tokens=-1,
            )

    def test_usage_is_immutable(self) -> None:
        usage = LLMUsage(
            prompt_tokens=10,
        )

        with pytest.raises(ValidationError):
            usage.prompt_tokens = 20


class TestLLMResponse:
    def test_valid_response(self) -> None:
        model = build_model_info()

        response = LLMResponse(
            text="RAG retrieves relevant knowledge before generation.",
            model=model,
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
            finish_reason="stop",
        )

        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )
        assert response.model == model
        assert response.usage is not None
        assert response.usage.total_tokens == 120
        assert response.finish_reason == "stop"

    def test_response_without_usage_is_valid(self) -> None:
        response = LLMResponse(
            text="Generated response.",
            model=build_model_info(),
        )

        assert response.usage is None
        assert response.finish_reason is None
        assert response.metadata == {}

    def test_blank_response_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(
                text="   ",
                model=build_model_info(),
            )

    def test_blank_finish_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(
                text="Generated response.",
                model=build_model_info(),
                finish_reason="   ",
            )

    def test_response_is_immutable(self) -> None:
        response = LLMResponse(
            text="Generated response.",
            model=build_model_info(),
        )

        with pytest.raises(ValidationError):
            response.text = "Changed response."


class TestLLMContracts:
    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMMessage(
                role="user",
                content="Test.",
                unexpected_field=True,
            )

    def test_request_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[
                    LLMMessage(
                        role="user",
                        content="Test.",
                    ),
                ],
                model=build_model_info(),
                unexpected_field=True,
            )

    def test_response_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(
                text="Test response.",
                model=build_model_info(),
                unexpected_field=True,
            )