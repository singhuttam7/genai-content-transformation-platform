from __future__ import annotations

import asyncio

import pytest

from app.ingestion.video.providers.local.parsing import (
    VisionResponseParser,
)
from app.ingestion.video.providers.local.provider import (
    LocalVisionProvider,
)
from app.ingestion.video.providers.local.runtime import (
    VisionRuntimeRequest,
    VisionRuntimeResponse,
)
from app.ingestion.video.schemas import (
    VideoFrame,
    VisionRequest,
    VisionStatus,
)


# ============================================================================
# Fake runtime
# ============================================================================


class FakeVisionRuntime:
    """
    Deterministic fake runtime used to test LocalVisionProvider.

    This runtime never calls Ollama or any external service.
    """

    name = "fake-runtime"

    def __init__(
        self,
        response_text: str,
        *,
        model: str = "fake-model",
        metadata: dict | None = None,
    ) -> None:
        self.response_text = response_text
        self.model = model
        self.metadata = metadata or {}

        self.requests: list[VisionRuntimeRequest] = []

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        """
        Record the runtime request and return a deterministic response.
        """

        self.requests.append(request)

        return VisionRuntimeResponse(
            text=self.response_text,
            model=self.model,
            runtime_name=self.name,
            metadata=self.metadata,
        )


# ============================================================================
# Test helpers
# ============================================================================


def make_frame(
    *,
    frame_index: int,
    timestamp_seconds: float,
) -> VideoFrame:
    """
    Create a minimal valid JPEG-like frame for unit testing.
    """

    return VideoFrame(
        timestamp_seconds=timestamp_seconds,
        frame_index=frame_index,
        image=b"\xff\xd8\xff\xd9",
    )


def make_request(
    frames: list[VideoFrame],
    *,
    prompt: str | None = None,
    detail_level: str = "standard",
) -> VisionRequest:
    """
    Create a VisionRequest for provider tests.
    """

    return VisionRequest(
        frames=frames,
        prompt=prompt,
        detail_level=detail_level,
    )


# ============================================================================
# Successful structured analysis
# ============================================================================


@pytest.mark.asyncio
async def test_provider_returns_completed_result_for_structured_response():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"timestamp_seconds": 0.0,'
            '"frame_index": 0,'
            '"description": "A person is standing indoors.",'
            '"objects": ["person"],'
            '"entities": [],'
            '"actions": ["standing"],'
            '"scene": "indoor room",'
            '"visible_text": null,'
            '"confidence": 0.95'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=2.5,
            )
        ]
    )

    result = await provider.analyze(request)

    assert result.status == VisionStatus.COMPLETED
    assert result.requested_frames == 1
    assert result.processed_frames == 1
    assert result.failed_frames == 0

    assert len(result.observations) == 1

    assert (
        result.observations[0].description
        == "A person is standing indoors."
    )


# ============================================================================
# Multiple observations
# ============================================================================


@pytest.mark.asyncio
async def test_provider_handles_multiple_observations():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"timestamp_seconds": 0.0,'
            '"frame_index": 0,'
            '"description": "A car is parked.",'
            '"objects": ["car"]'
            "},"
            "{"
            '"timestamp_seconds": 2.0,'
            '"frame_index": 1,'
            '"description": "The car is moving.",'
            '"objects": ["car"],'
            '"actions": ["moving"]'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
            ),
            make_frame(
                frame_index=1,
                timestamp_seconds=3.0,
            ),
        ]
    )

    result = await provider.analyze(request)

    assert result.status == VisionStatus.COMPLETED
    assert result.requested_frames == 2
    assert result.processed_frames == 2
    assert result.failed_frames == 0

    assert len(result.observations) == 2

    assert (
        result.observations[0].description
        == "A car is parked."
    )

    assert (
        result.observations[1].description
        == "The car is moving."
    )


# ============================================================================
# Frame metadata restoration
# ============================================================================


@pytest.mark.asyncio
async def test_provider_restores_authoritative_frame_index():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"timestamp_seconds": 999.0,'
            '"frame_index": 999,'
            '"description": "A test frame."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=7,
                timestamp_seconds=12.75,
            )
        ]
    )

    result = await provider.analyze(request)

    assert len(result.observations) == 1

    observation = result.observations[0]

    assert observation.frame_index == 7


@pytest.mark.asyncio
async def test_provider_restores_authoritative_timestamp():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"timestamp_seconds": 999.0,'
            '"frame_index": 0,'
            '"description": "A test frame."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=18.375,
            )
        ]
    )

    result = await provider.analyze(request)

    assert len(result.observations) == 1

    observation = result.observations[0]

    assert observation.timestamp_seconds == 18.375


# ============================================================================
# Partial result
# ============================================================================


@pytest.mark.asyncio
async def test_provider_returns_partial_when_not_all_frames_are_observed():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"timestamp_seconds": 0.0,'
            '"frame_index": 0,'
            '"description": "Only the first frame was analyzed."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
            ),
            make_frame(
                frame_index=1,
                timestamp_seconds=3.0,
            ),
        ]
    )

    result = await provider.analyze(request)

    assert result.status == VisionStatus.PARTIAL
    assert result.requested_frames == 2
    assert result.processed_frames == 1
    assert result.failed_frames == 1


# ============================================================================
# No frames
# ============================================================================


@pytest.mark.asyncio
async def test_provider_returns_no_frames_without_calling_runtime():
    runtime = FakeVisionRuntime(
        response_text='{"observations": []}'
    )

    provider = LocalVisionProvider(runtime)

    request = make_request([])

    result = await provider.analyze(request)

    assert result.status == VisionStatus.NO_FRAMES
    assert result.requested_frames == 0
    assert result.processed_frames == 0
    assert result.failed_frames == 0

    assert runtime.requests == []


# ============================================================================
# Prompt construction
# ============================================================================


@pytest.mark.asyncio
async def test_provider_passes_structured_prompt_to_runtime():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Test frame."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=4.0,
            )
        ],
        prompt="Focus on security-relevant objects.",
        detail_level="high",
    )

    await provider.analyze(request)

    assert len(runtime.requests) == 1

    runtime_request = runtime.requests[0]

    assert runtime_request.system_prompt
    assert runtime_request.user_prompt

    assert (
        "security-relevant objects"
        in runtime_request.user_prompt
    )

    assert "JSON" in runtime_request.user_prompt


# ============================================================================
# Frame serialization
# ============================================================================


@pytest.mark.asyncio
async def test_provider_serializes_frames_before_runtime_invocation():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Test frame."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=2.0,
            )
        ]
    )

    await provider.analyze(request)

    assert len(runtime.requests) == 1

    runtime_request = runtime.requests[0]

    assert len(runtime_request.frames) == 1

    serialized_frame = runtime_request.frames[0]

    assert serialized_frame.frame_index == 0
    assert serialized_frame.timestamp_seconds == 2.0
    assert serialized_frame.mime_type == "image/jpeg"
    assert serialized_frame.data


# ============================================================================
# Runtime metadata propagation
# ============================================================================


@pytest.mark.asyncio
async def test_provider_preserves_runtime_metadata():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Test frame."'
            "}"
            "]}"
        ),
        model="gemma3:4b",
        metadata={
            "total_duration": 123456,
            "load_duration": 1234,
        },
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
            )
        ]
    )

    result = await provider.analyze(request)

    assert result.metadata["provider"] == "local"
    assert result.metadata["runtime"] == "fake-runtime"
    assert result.metadata["model"] == "gemma3:4b"

    assert (
        result.metadata["runtime_metadata"]["total_duration"]
        == 123456
    )

    assert (
        result.metadata["runtime_metadata"]["load_duration"]
        == 1234
    )


# ============================================================================
# Parser metadata propagation
# ============================================================================


@pytest.mark.asyncio
async def test_provider_preserves_parser_metadata():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Test frame."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(
        runtime,
        response_parser=VisionResponseParser(),
    )

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
            )
        ]
    )

    result = await provider.analyze(request)

    assert result.metadata["parser"]
    assert result.metadata["source_format"]


# ============================================================================
# Runtime independence
# ============================================================================


@pytest.mark.asyncio
async def test_provider_does_not_depend_on_ollama_runtime():
    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Runtime-independent result."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=5.0,
            )
        ]
    )

    result = await provider.analyze(request)

    assert result.status == VisionStatus.COMPLETED

    assert (
        result.observations[0].description
        == "Runtime-independent result."
    )


# ============================================================================
# Dependency injection
# ============================================================================


@pytest.mark.asyncio
async def test_provider_accepts_custom_prompt_builder():
    class CustomPromptBuilder:
        name = "custom"

        def build(self, request: VisionRequest):
            from app.ingestion.video.providers.local.prompting import (
                VisionPrompt,
            )

            return VisionPrompt(
                system_prompt="CUSTOM SYSTEM",
                user_prompt="CUSTOM USER",
            )

    runtime = FakeVisionRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Custom prompt result."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(
        runtime,
        prompt_builder=CustomPromptBuilder(),
    )

    request = make_request(
        [
            make_frame(
                frame_index=0,
                timestamp_seconds=1.0,
            )
        ]
    )

    await provider.analyze(request)

    assert len(runtime.requests) == 1

    runtime_request = runtime.requests[0]

    assert runtime_request.system_prompt == "CUSTOM SYSTEM"
    assert runtime_request.user_prompt == "CUSTOM USER"


# ============================================================================
# Dependency validation
# ============================================================================


def test_provider_rejects_missing_runtime():
    with pytest.raises(
        ValueError,
        match="runtime",
    ):
        LocalVisionProvider(None)


def test_provider_rejects_missing_request():
    runtime = FakeVisionRuntime(
        response_text='{"observations": []}'
    )

    provider = LocalVisionProvider(runtime)

    with pytest.raises(
        ValueError,
        match="request",
    ):
        asyncio.run(
            provider.analyze(None)
        )