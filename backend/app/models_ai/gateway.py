from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.models_ai.llm.errors import (
    LLMResponseError,
    LLMStructuredOutputError,
)
from app.models_ai.llm.port import LLMPort
from app.models_ai.llm.schemas import (
    LLMRequest,
    LLMResponse,
)
from app.models_ai.llm.structured import StructuredOutputParser


StructuredModelT = TypeVar(
    "StructuredModelT",
    bound=BaseModel,
)


class LLMGateway:
    """
    Application-facing gateway for provider-independent LLM generation.

    The gateway depends only on LLMPort and therefore remains
    independent of any concrete LLM provider.
    """

    def __init__(
        self,
        port: LLMPort,
    ) -> None:
        self._port = port

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate an LLM response through the configured port.

        Provider-independent LLM errors are propagated unchanged.
        Unexpected provider responses are converted into
        LLMResponseError.
        """

        response = await self._port.generate(request)

        if not isinstance(response, LLMResponse):
            provider = request.model.provider

            raise LLMResponseError(
                "LLM provider returned an invalid response type.",
                provider=provider,
            )

        return response

    async def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredModelT],
    ) -> StructuredModelT:
        """
        Generate an LLM response and validate it as a Pydantic model.

        The provider remains responsible only for text generation.
        JSON decoding and schema validation happen at the gateway
        boundary in a provider-independent manner.
        """

        response = await self.generate(request)

        try:
            return StructuredOutputParser.parse(
                text=response.text,
                response_model=response_model,
            )
        except ValueError as exc:
            raise LLMStructuredOutputError(
                str(exc),
                provider=request.model.provider,
            ) from exc