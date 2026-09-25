from __future__ import annotations

from abc import abstractmethod

from app.models_ai.llm.errors import LLMProviderError
from app.models_ai.llm.port import LLMPort
from app.models_ai.llm.schemas import (
    LLMRequest,
    LLMResponse,
)
from app.models_ai.providers.schemas import (
    LLMProviderConfig,
)


class LLMProviderAdapter(LLMPort):
    """
    Base class for concrete LLM provider adapters.

    Provider implementations inherit from this class and implement
    provider-specific generation and error translation.
    """

    def __init__(
        self,
        config: LLMProviderConfig,
    ) -> None:
        self._config = config

    @property
    def config(self) -> LLMProviderConfig:
        """Return the immutable provider configuration."""

        return self._config

    @property
    def provider(self) -> str:
        """Return the provider identifier."""

        return self._config.provider

    def validate_request(
        self,
        request: LLMRequest,
    ) -> None:
        """
        Validate provider-level request compatibility.

        Concrete adapters can override this method when a provider
        imposes additional constraints.
        """

        if request.model.provider != self.provider:
            raise LLMProviderError(
                (
                    "LLM request provider does not match "
                    "the configured adapter provider."
                ),
                provider=self.provider,
            )

    @abstractmethod
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response using the concrete provider.

        Implementations must translate provider-specific behavior
        into the platform's LLMResponse contract.
        """

        raise NotImplementedError