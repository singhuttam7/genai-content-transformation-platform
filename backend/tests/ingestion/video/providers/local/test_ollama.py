import json

import httpx
import pytest

from app.ingestion.video.providers.local.config import (
    LocalVisionConfig,
)
from app.ingestion.video.providers.local.ollama import (
    OllamaVisionRuntime,
    ensure_vision_runtime,
)
from app.ingestion.video.providers.local.runtime import (
    VisionRuntime,
    VisionRuntimeRequest,
    VisionRuntimeResponse,
)
from app.ingestion.video.providers.local.serialization import (
    SerializedFrame,
)


def make_config() -> LocalVisionConfig:
    return LocalVisionConfig(
        model="gemma3:4b",
        base_url="http://localhost:11434",
        timeout_seconds=30.0,
        max_retries=1,
        temperature=0.1,
        keep_alive="5m",
    )


def make_frame(
    *,
    frame_index: int = 1,
    timestamp_seconds: float = 2.0,
) -> SerializedFrame:
    return SerializedFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        mime_type="image/jpeg",
        data="YWJj",
    )


def make_request() -> VisionRuntimeRequest:
    return VisionRuntimeRequest(
        model="gemma3:4b",
        system_prompt="You are a visual analysis component.",
        user_prompt="Describe the supplied frame.",
        frames=(make_frame(),),
        temperature=0.1,
        keep_alive="5m",
    )


class TestOllamaVisionRuntime:
    def test_runtime_name(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        assert runtime.name == "ollama"

    def test_chat_path(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        assert runtime.CHAT_PATH == "/api/chat"

    def test_configuration_is_preserved(self) -> None:
        config = make_config()

        runtime = OllamaVisionRuntime(config)

        assert runtime.config is config

    def test_client_defaults_to_none(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        assert runtime.client is None

    @pytest.mark.asyncio
    async def test_custom_client_is_preserved(self) -> None:
        client = httpx.AsyncClient()

        try:
            runtime = OllamaVisionRuntime(
                make_config(),
                client=client,
            )

            assert runtime.client is client

        finally:
            await client.aclose()

    def test_runtime_implements_protocol_shape(self) -> None:
        runtime: VisionRuntime = OllamaVisionRuntime(
            make_config(),
        )

        assert callable(runtime.generate)

    def test_build_payload_contains_model(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        payload = runtime._build_payload(
            make_request(),
        )

        assert payload["model"] == "gemma3:4b"

    def test_build_payload_disables_streaming(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        payload = runtime._build_payload(
            make_request(),
        )

        assert payload["stream"] is False

    def test_build_payload_contains_messages(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        payload = runtime._build_payload(
            make_request(),
        )

        messages = payload["messages"]

        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_build_payload_contains_system_message(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        request = make_request()

        payload = runtime._build_payload(request)

        messages = payload["messages"]

        assert messages[0] == {
            "role": "system",
            "content": (
                "You are a visual analysis component."
            ),
        }

    def test_build_payload_contains_user_message(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        request = make_request()

        payload = runtime._build_payload(request)

        messages = payload["messages"]

        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == (
            "Describe the supplied frame."
        )

    def test_build_payload_contains_serialized_images(
        self,
    ) -> None:
        runtime = OllamaVisionRuntime(make_config())

        request = make_request()

        payload = runtime._build_payload(request)

        messages = payload["messages"]

        assert messages[1]["images"] == ["YWJj"]

    def test_build_payload_contains_temperature(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        payload = runtime._build_payload(
            make_request(),
        )

        assert payload["options"]["temperature"] == 0.1

    def test_build_payload_contains_keep_alive(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        payload = runtime._build_payload(
            make_request(),
        )

        assert payload["keep_alive"] == "5m"

    def test_build_payload_omits_keep_alive_when_none(
        self,
    ) -> None:
        config = LocalVisionConfig(
            keep_alive=None,
        )

        runtime = OllamaVisionRuntime(config)

        request = VisionRuntimeRequest(
            model="gemma3:4b",
            system_prompt="System.",
            user_prompt="User.",
            keep_alive=None,
        )

        payload = runtime._build_payload(request)

        assert "keep_alive" not in payload

    def test_build_runtime_response(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        response = runtime._build_runtime_response(
            request=make_request(),
            data={
                "model": "gemma3:4b",
                "message": {
                    "role": "assistant",
                    "content": "A person is visible.",
                },
                "done": True,
            },
        )

        assert isinstance(
            response,
            VisionRuntimeResponse,
        )

        assert response.text == (
            "A person is visible."
        )

        assert response.model == "gemma3:4b"

        assert response.runtime_name == "ollama"

        assert response.metadata["runtime"] == "ollama"

        assert response.metadata["done"] is True

    def test_response_model_falls_back_to_request_model(
        self,
    ) -> None:
        runtime = OllamaVisionRuntime(make_config())

        response = runtime._build_runtime_response(
            request=make_request(),
            data={
                "message": {
                    "role": "assistant",
                    "content": "A scene is visible.",
                },
            },
        )

        assert response.model == "gemma3:4b"

    def test_invalid_response_payload_is_rejected(
        self,
    ) -> None:
        runtime = OllamaVisionRuntime(make_config())

        with pytest.raises(
            ValueError,
            match="invalid response payload",
        ):
            runtime._build_runtime_response(
                request=make_request(),
                data=[],
            )

    def test_missing_message_is_rejected(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        with pytest.raises(
            ValueError,
            match="valid message",
        ):
            runtime._build_runtime_response(
                request=make_request(),
                data={
                    "model": "gemma3:4b",
                },
            )

    def test_missing_response_text_is_rejected(
        self,
    ) -> None:
        runtime = OllamaVisionRuntime(make_config())

        with pytest.raises(
            ValueError,
            match="does not contain valid text",
        ):
            runtime._build_runtime_response(
                request=make_request(),
                data={
                    "model": "gemma3:4b",
                    "message": {
                        "role": "assistant",
                    },
                },
            )

    def test_empty_response_text_is_rejected(
        self,
    ) -> None:
        runtime = OllamaVisionRuntime(make_config())

        with pytest.raises(
            ValueError,
            match="does not contain valid text",
        ):
            runtime._build_runtime_response(
                request=make_request(),
                data={
                    "model": "gemma3:4b",
                    "message": {
                        "role": "assistant",
                        "content": "   ",
                    },
                },
            )

    @pytest.mark.asyncio
    async def test_none_request_is_rejected(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        with pytest.raises(
            ValueError,
            match="request must be provided",
        ):
            await runtime.generate(
                None,  # type: ignore[arg-type]
            )

    def test_runtime_contract_validation(self) -> None:
        runtime = OllamaVisionRuntime(make_config())

        validated = ensure_vision_runtime(runtime)

        assert validated is runtime

    def test_invalid_runtime_is_rejected(self) -> None:
        with pytest.raises(
            TypeError,
            match=(
                "does not implement the "
                "VisionRuntime contract"
            ),
        ):
            ensure_vision_runtime(object())


class TestOllamaHTTPBoundary:
    @pytest.mark.asyncio
    async def test_generate_translates_request_and_response(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        async def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)

            captured["json"] = json.loads(
                request.content,
            )

            return httpx.Response(
                status_code=200,
                json={
                    "model": "gemma3:4b",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "A person is standing "
                            "in a room."
                        ),
                    },
                    "done": True,
                    "eval_count": 10,
                },
            )

        transport = httpx.MockTransport(handler)

        async with httpx.AsyncClient(
            base_url="http://test-runtime",
            transport=transport,
        ) as client:
            runtime = OllamaVisionRuntime(
                make_config(),
                client=client,
            )

            response = await runtime.generate(
                make_request(),
            )

        assert response.text == (
            "A person is standing in a room."
        )

        assert response.model == "gemma3:4b"

        assert response.runtime_name == "ollama"

        assert captured["method"] == "POST"

        assert captured["url"] == (
            "http://test-runtime/api/chat"
        )

        payload = captured["json"]

        assert isinstance(payload, dict)

        assert payload["model"] == "gemma3:4b"

        assert payload["stream"] is False

        messages = payload["messages"]

        assert isinstance(messages, list)

        assert len(messages) == 2

        assert messages[0] == {
            "role": "system",
            "content": (
                "You are a visual analysis component."
            ),
        }

        assert messages[1]["role"] == "user"

        assert messages[1]["content"] == (
            "Describe the supplied frame."
        )

        assert messages[1]["images"] == [
            "YWJj",
        ]

        assert payload["options"]["temperature"] == 0.1

        assert payload["keep_alive"] == "5m"

    @pytest.mark.asyncio
    async def test_http_error_is_propagated(
        self,
    ) -> None:
        async def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            return httpx.Response(
                status_code=500,
                json={
                    "error": "model failure",
                },
            )

        transport = httpx.MockTransport(handler)

        async with httpx.AsyncClient(
            base_url="http://test-runtime",
            transport=transport,
        ) as client:
            runtime = OllamaVisionRuntime(
                make_config(),
                client=client,
            )

            with pytest.raises(
                httpx.HTTPStatusError,
            ):
                await runtime.generate(
                    make_request(),
                )

    @pytest.mark.asyncio
    async def test_malformed_json_response_is_rejected(
        self,
    ) -> None:
        async def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b"not-json",
            )

        transport = httpx.MockTransport(handler)

        async with httpx.AsyncClient(
            base_url="http://test-runtime",
            transport=transport,
        ) as client:
            runtime = OllamaVisionRuntime(
                make_config(),
                client=client,
            )

            with pytest.raises(
                ValueError,
                match="invalid JSON response",
            ):
                await runtime.generate(
                    make_request(),
                )

    @pytest.mark.asyncio
    async def test_external_client_is_not_closed_by_runtime(
        self,
    ) -> None:
        async def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                json={
                    "model": "gemma3:4b",
                    "message": {
                        "role": "assistant",
                        "content": "Description.",
                    },
                    "done": True,
                },
            )

        transport = httpx.MockTransport(handler)

        client = httpx.AsyncClient(
            base_url="http://test-runtime",
            transport=transport,
        )

        runtime = OllamaVisionRuntime(
            make_config(),
            client=client,
        )

        await runtime.generate(
            make_request(),
        )

        assert client.is_closed is False

        await client.aclose()