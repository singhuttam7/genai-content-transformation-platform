from __future__ import annotations

from typing import Any

import httpx

from app.ingestion.video.providers.local.config import LocalVisionConfig
from app.ingestion.video.providers.local.runtime import (
    VisionRuntime,
    VisionRuntimeRequest,
    VisionRuntimeResponse,
)


class OllamaVisionRuntime:
    """
    Ollama implementation of the provider-independent VisionRuntime.

    This adapter is the only layer in the local vision subsystem
    that knows about the Ollama HTTP API.

    Higher-level services depend on the provider-independent
    VisionRuntime contract instead of depending directly on Ollama.
    """

    name = "ollama"

    CHAT_PATH = "/api/chat"

    def __init__(
        self,
        config: LocalVisionConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config is None:
            raise ValueError(
                "Ollama runtime configuration must be provided."
            )

        self._config = config
        self._client = client
        self._owns_client = client is None

    @property
    def config(self) -> LocalVisionConfig:
        """
        Return the configured Ollama runtime configuration.
        """

        return self._config

    @property
    def client(self) -> httpx.AsyncClient | None:
        """
        Return the configured HTTP client.

        This is primarily useful for lifecycle management
        and testing.
        """

        return self._client

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        """
        Execute a multimodal vision request through Ollama.

        The provider-independent VisionRuntimeRequest is translated
        into Ollama's /api/chat multimodal request format.
        """

        if request is None:
            raise ValueError(
                "Vision runtime request must be provided."
            )

        client = await self._get_client()

        payload = self._build_payload(request)

        response = await client.post(
            self.CHAT_PATH,
            json=payload,
        )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(
                "Ollama returned an invalid JSON response."
            ) from exc

        return self._build_runtime_response(
            request=request,
            data=data,
        )

    async def close(self) -> None:
        """
        Close the internally owned HTTP client.

        An externally supplied client remains owned by its caller.
        """

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Lazily create and return the HTTP client.

        The client is reused across multiple inference requests.
        """

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )

        return self._client

    def _build_payload(
        self,
        request: VisionRuntimeRequest,
    ) -> dict[str, Any]:
        """
        Translate a provider-independent vision request into
        the Ollama /api/chat multimodal request format.
        """

        images = [
            frame.data
            for frame in request.frames
        ]

        messages = [
            {
                "role": "system",
                "content": request.system_prompt,
            },
            {
                "role": "user",
                "content": request.user_prompt,
                "images": images,
            },
        ]

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive

        return payload

    def _build_runtime_response(
        self,
        *,
        request: VisionRuntimeRequest,
        data: object,
    ) -> VisionRuntimeResponse:
        """
        Translate an Ollama /api/chat response into the
        provider-independent VisionRuntimeResponse contract.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Ollama returned an invalid response payload."
            )

        message = data.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Ollama response does not contain a valid message."
            )

        response_text = message.get("content")

        if not isinstance(response_text, str):
            raise ValueError(
                "Ollama response does not contain valid text."
            )

        if not response_text.strip():
            raise ValueError(
                "Ollama response does not contain valid text."
            )

        response_model = data.get("model")

        if (
            not isinstance(response_model, str)
            or not response_model.strip()
        ):
            response_model = request.model

        metadata: dict[str, object] = {
            "runtime": self.name,
        }

        for key in (
            "created_at",
            "done",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            if key in data:
                metadata[key] = data[key]

        return VisionRuntimeResponse(
            text=response_text,
            model=response_model,
            runtime_name=self.name,
            metadata=metadata,
        )


def ensure_vision_runtime(
    runtime: object,
) -> VisionRuntime:
    """
    Validate that an object satisfies the expected VisionRuntime
    structural contract.

    This helper deliberately does not require a specific runtime
    implementation.
    """

    generate = getattr(runtime, "generate", None)

    if not callable(generate):
        raise TypeError(
            "Configured object does not implement the "
            "VisionRuntime contract."
        )

    return runtime  # type: ignore[return-value]