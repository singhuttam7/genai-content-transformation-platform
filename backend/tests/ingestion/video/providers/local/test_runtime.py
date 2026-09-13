import pytest

from app.ingestion.video.providers.local.runtime import (
    VisionRuntime,
    VisionRuntimeRequest,
    VisionRuntimeResponse,
)
from app.ingestion.video.providers.local.serialization import (
    SerializedFrame,
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


class FakeVisionRuntime:
    name = "fake-runtime"

    def __init__(
        self,
        response: VisionRuntimeResponse,
    ) -> None:
        self.response = response
        self.received_request: VisionRuntimeRequest | None = None

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        self.received_request = request
        return self.response


class TestVisionRuntimeRequest:
    def test_default_values(self) -> None:
        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System instruction.",
            user_prompt="Analyze the image.",
        )

        assert request.model == "test-model"
        assert request.system_prompt == "System instruction."
        assert request.user_prompt == "Analyze the image."
        assert request.frames == ()
        assert request.temperature == 0.1
        assert request.keep_alive == "5m"
        assert request.metadata == {}

    def test_custom_values_are_preserved(self) -> None:
        frame = make_frame()

        request = VisionRuntimeRequest(
            model="vision-model",
            system_prompt="System.",
            user_prompt="User.",
            frames=(frame,),
            temperature=0.7,
            keep_alive="10m",
            metadata={"source": "test"},
        )

        assert request.model == "vision-model"
        assert request.frames == (frame,)
        assert request.temperature == 0.7
        assert request.keep_alive == "10m"
        assert request.metadata == {"source": "test"}

    def test_model_is_trimmed(self) -> None:
        request = VisionRuntimeRequest(
            model="  test-model  ",
            system_prompt="System.",
            user_prompt="User.",
        )

        assert request.model == "test-model"

    def test_prompts_are_trimmed(self) -> None:
        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="  System.  ",
            user_prompt="  User.  ",
        )

        assert request.system_prompt == "System."
        assert request.user_prompt == "User."

    def test_frames_are_converted_to_tuple(self) -> None:
        frames = [
            make_frame(frame_index=1),
            make_frame(frame_index=2),
        ]

        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System.",
            user_prompt="User.",
            frames=frames,
        )

        assert isinstance(request.frames, tuple)
        assert request.frames == tuple(frames)

    def test_empty_model_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="model must not be empty",
        ):
            VisionRuntimeRequest(
                model="   ",
                system_prompt="System.",
                user_prompt="User.",
            )

    def test_empty_system_prompt_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="system prompt must not be empty",
        ):
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="   ",
                user_prompt="User.",
            )

    def test_empty_user_prompt_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="user prompt must not be empty",
        ):
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="System.",
                user_prompt="   ",
            )

    def test_temperature_below_range_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="temperature must be between 0.0 and 2.0",
        ):
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="System.",
                user_prompt="User.",
                temperature=-0.1,
            )

    def test_temperature_above_range_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="temperature must be between 0.0 and 2.0",
        ):
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="System.",
                user_prompt="User.",
                temperature=2.1,
            )

    def test_temperature_boundaries_are_allowed(self) -> None:
        assert (
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="System.",
                user_prompt="User.",
                temperature=0.0,
            ).temperature
            == 0.0
        )

        assert (
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="System.",
                user_prompt="User.",
                temperature=2.0,
            ).temperature
            == 2.0
        )

    def test_none_keep_alive_is_allowed(self) -> None:
        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System.",
            user_prompt="User.",
            keep_alive=None,
        )

        assert request.keep_alive is None

    def test_empty_keep_alive_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="keep-alive must not be empty",
        ):
            VisionRuntimeRequest(
                model="test-model",
                system_prompt="System.",
                user_prompt="User.",
                keep_alive="   ",
            )

    def test_request_is_immutable(self) -> None:
        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System.",
            user_prompt="User.",
        )

        with pytest.raises(AttributeError):
            request.model = "another-model"


class TestVisionRuntimeResponse:
    def test_response_values_are_preserved(self) -> None:
        response = VisionRuntimeResponse(
            text="A person is visible.",
            model="vision-model",
            runtime_name="fake-runtime",
            metadata={"latency_ms": 12.0},
        )

        assert response.text == "A person is visible."
        assert response.model == "vision-model"
        assert response.runtime_name == "fake-runtime"
        assert response.metadata == {"latency_ms": 12.0}

    def test_response_text_is_trimmed(self) -> None:
        response = VisionRuntimeResponse(
            text="  A person is visible.  ",
            model="vision-model",
            runtime_name="fake-runtime",
        )

        assert response.text == "A person is visible."

    def test_response_model_is_trimmed(self) -> None:
        response = VisionRuntimeResponse(
            text="Description.",
            model="  vision-model  ",
            runtime_name="fake-runtime",
        )

        assert response.model == "vision-model"

    def test_runtime_name_is_trimmed(self) -> None:
        response = VisionRuntimeResponse(
            text="Description.",
            model="vision-model",
            runtime_name="  fake-runtime  ",
        )

        assert response.runtime_name == "fake-runtime"

    def test_empty_text_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="response text must not be empty",
        ):
            VisionRuntimeResponse(
                text="   ",
                model="vision-model",
                runtime_name="fake-runtime",
            )

    def test_empty_model_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="response model must not be empty",
        ):
            VisionRuntimeResponse(
                text="Description.",
                model="   ",
                runtime_name="fake-runtime",
            )

    def test_empty_runtime_name_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="runtime name must not be empty",
        ):
            VisionRuntimeResponse(
                text="Description.",
                model="vision-model",
                runtime_name="   ",
            )

    def test_response_is_immutable(self) -> None:
        response = VisionRuntimeResponse(
            text="Description.",
            model="vision-model",
            runtime_name="fake-runtime",
        )

        with pytest.raises(AttributeError):
            response.text = "Changed."


class TestVisionRuntimeProtocol:
    @pytest.mark.asyncio
    async def test_protocol_can_be_implemented(self) -> None:
        response = VisionRuntimeResponse(
            text="A scene is visible.",
            model="test-model",
            runtime_name="fake-runtime",
        )

        runtime: VisionRuntime = FakeVisionRuntime(response)

        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System.",
            user_prompt="Analyze.",
            frames=(make_frame(),),
        )

        result = await runtime.generate(request)

        assert result is response

    @pytest.mark.asyncio
    async def test_runtime_receives_request(self) -> None:
        response = VisionRuntimeResponse(
            text="A scene is visible.",
            model="test-model",
            runtime_name="fake-runtime",
        )

        runtime = FakeVisionRuntime(response)

        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System.",
            user_prompt="Analyze.",
            frames=(
                make_frame(frame_index=4),
                make_frame(frame_index=5),
            ),
        )

        await runtime.generate(request)

        assert runtime.received_request is request
        assert runtime.received_request.frames[0].frame_index == 4
        assert runtime.received_request.frames[1].frame_index == 5

    def test_request_contains_no_runtime_specific_fields(self) -> None:
        request = VisionRuntimeRequest(
            model="test-model",
            system_prompt="System.",
            user_prompt="User.",
        )

        assert set(request.__dataclass_fields__) == {
            "model",
            "system_prompt",
            "user_prompt",
            "frames",
            "temperature",
            "keep_alive",
            "metadata",
        }

    def test_response_contains_no_runtime_specific_fields(self) -> None:
        response = VisionRuntimeResponse(
            text="Description.",
            model="test-model",
            runtime_name="fake-runtime",
        )

        assert set(response.__dataclass_fields__) == {
            "text",
            "model",
            "runtime_name",
            "metadata",
        }