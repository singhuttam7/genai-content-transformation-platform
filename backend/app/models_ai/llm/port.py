from __future__ import annotations

from abc import ABC, abstractmethod

from app.models_ai.llm.schemas import (
    LLMRequest,
    LLMResponse,
)


class LLMPort(ABC):
    """
    Provider-independent interface for LLM generation.

    Concrete provider adapters must implement this interface.
    """

    @abstractmethod
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response for the supplied LLM request.

        Implementations must translate provider-specific behavior
        into the provider-independent LLMRequest/LLMResponse contracts.
        """

        raise NotImplementedError