from __future__ import annotations

import pytest

from app.models_ai.llm.port import LLMPort
from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
)


def build_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Explain RAG.",
            ),
        ],
        model=LLMModelInfo(
            provider="test-provider",
            model_name="test-model",
        ),
    )


def build_response() -> LLMResponse:
    return LLMResponse(
        text="RAG retrieves relevant knowledge before generation.",
        model=LLMModelInfo(
            provider="test-provider",
            model_name="test-model",
        ),
    )


class TestLLMPort:
    def test_port_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            LLMPort()

    def test_concrete_adapter_can_implement_port(self) -> None:
        class FakeLLMAdapter(LLMPort):
            async def generate(
                self,
                request: LLMRequest,
            ) -> LLMResponse:
                return build_response()

        adapter = FakeLLMAdapter()

        assert isinstance(adapter, LLMPort)

    @pytest.mark.asyncio
    async def test_concrete_adapter_generate_contract(self) -> None:
        class FakeLLMAdapter(LLMPort):
            async def generate(
                self,
                request: LLMRequest,
            ) -> LLMResponse:
                assert isinstance(request, LLMRequest)

                return build_response()

        adapter = FakeLLMAdapter()

        response = await adapter.generate(
            build_request(),
        )

        assert isinstance(response, LLMResponse)
        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )