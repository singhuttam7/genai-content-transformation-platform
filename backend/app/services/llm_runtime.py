from __future__ import annotations

from app.core.config import settings
from app.models_ai.gateway import LLMGateway
from app.models_ai.llm import LLMModelInfo
from app.models_ai.providers import GroqLLMAdapter
from app.models_ai.providers.schemas import LLMProviderConfig


def get_llm_model() -> LLMModelInfo:
    """
    Build the provider-independent model description used by agents.
    """

    if not settings.llm_model:
        raise RuntimeError(
            "LLM model is not configured. "
            "Set LLM_MODEL in the backend environment."
        )

    return LLMModelInfo(
        provider=settings.llm_provider,
        model_name=settings.llm_model,
    )


def create_llm_gateway() -> LLMGateway:
    """
    Construct the application LLM gateway from configured settings.
    """

    provider = settings.llm_provider.strip().lower()

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "Groq API key is not configured. "
                "Set GROQ_API_KEY in the backend environment."
            )

        provider_config = LLMProviderConfig(
            provider="groq",
            api_key=settings.groq_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            default_model=settings.llm_model or None,
        )

        adapter = GroqLLMAdapter(
            provider_config,
        )

        return LLMGateway(adapter)

    raise RuntimeError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )