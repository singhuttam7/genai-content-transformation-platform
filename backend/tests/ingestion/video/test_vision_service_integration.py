from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.ingestion.video.provider import VisionProvider
from app.ingestion.video.schemas import (
    VideoFrame,
    VisionObservation,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from app.ingestion.video.vision import VisionService
from app.ingestion.video.providers.local.provider import (
    LocalVisionProvider,
)
from app.ingestion.video.providers.local.runtime import (
    VisionRuntimeRequest,
    VisionRuntimeResponse,
)


# ============================================================================
# Fake Vision Provider
# ============================================================================


class FakeVisionProvider:
    """
    Deterministic VisionProvider implementation used to test
    VisionService independently from any real model or runtime.
    """

    name = "fake-provider"

    def __init__(
        self,
        *,
        result: VisionResult | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.result = result
        self.exception = exception
        self.requests: list[VisionRequest] = []

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        self.requests.append(request)

        if self.exception is not None:
            raise self.exception

        if self.result is not None:
            return self.result

        return VisionResult(
            status=VisionStatus.COMPLETED,
            observations=[
                VisionObservation(
                    timestamp_seconds=1.0,
                    frame_index=0,
                    description="Fake observation.",
                )
            ],
            requested_frames=len(request.frames),
            processed_frames=1,
            failed_frames=max(
                len(request.frames) - 1,
                0,
            ),
            errors=[],
            metadata={
                "provider": self.name,
            },
        )


# ============================================================================
# Fake Local Runtime
# ============================================================================


class FakeLocalVisionRuntime:
    """
    Deterministic runtime for verifying that LocalVisionProvider
    can be integrated underneath VisionService.
    """

    name = "fake-local-runtime"

    def __init__(
        self,
        response_text: str | None = None,
    ) -> None:
        self.requests: list[VisionRuntimeRequest] = []

        self.response_text = (
            response_text
            if response_text is not None
            else (
                '{"observations": ['
                "{"
                '"frame_index": 0,'
                '"timestamp_seconds": 999.0,'
                '"description": "Local provider observation."'
                "}"
                "]}"
            )
        )

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        self.requests.append(request)

        return VisionRuntimeResponse(
            text=self.response_text,
            model="fake-local-model",
            runtime_name=self.name,
            metadata={
                "integration_test": True,
            },
        )


# ============================================================================
# Helpers
# ============================================================================


def make_frame(
    *,
    frame_index: int = 0,
    timestamp_seconds: float = 1.0,
) -> VideoFrame:
    """
    Create a minimal valid JPEG frame.
    """

    return VideoFrame(
        timestamp_seconds=timestamp_seconds,
        frame_index=frame_index,
        image=b"\xff\xd8\xff\xd9",
    )


def make_request(
    *,
    frame_count: int = 1,
    max_observations: int | None = None,
) -> VisionRequest:
    """
    Create a valid provider-independent VisionRequest.
    """

    return VisionRequest(
        frames=[
            make_frame(
                frame_index=index,
                timestamp_seconds=float(index + 1),
            )
            for index in range(frame_count)
        ],
        max_observations=max_observations,
    )


def make_completed_result(
    *,
    provider_name: str = "fake-provider",
    requested_frames: int = 1,
) -> VisionResult:
    """
    Create a deterministic successful VisionResult.
    """

    return VisionResult(
        status=VisionStatus.COMPLETED,
        observations=[
            VisionObservation(
                timestamp_seconds=1.0,
                frame_index=0,
                description="Completed observation.",
            )
        ],
        requested_frames=requested_frames,
        processed_frames=1,
        failed_frames=max(
            requested_frames - 1,
            0,
        ),
        errors=[],
        metadata={
            "provider": provider_name,
            "custom_metadata": "preserved",
        },
    )


# ============================================================================
# Constructor / dependency injection
# ============================================================================


def test_vision_service_requires_provider():
    with pytest.raises(
        ValueError,
        match="Vision provider must be provided",
    ):
        VisionService(None)  # type: ignore[arg-type]


def test_vision_service_exposes_injected_provider():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    assert service.provider is provider


def test_local_provider_can_be_injected_into_vision_service():
    runtime = FakeLocalVisionRuntime()

    provider = LocalVisionProvider(
        runtime,
    )

    service = VisionService(provider)

    assert service.provider is provider


# ============================================================================
# Provider invocation
# ============================================================================


@pytest.mark.asyncio
async def test_service_invokes_provider_exactly_once():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = make_request()

    await service.analyze(request)

    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_service_passes_provider_independent_request_to_provider():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = make_request(
        frame_count=2,
    )

    await service.analyze(request)

    assert len(provider.requests) == 1

    forwarded_request = provider.requests[0]

    assert isinstance(
        forwarded_request,
        VisionRequest,
    )

    assert len(forwarded_request.frames) == 2


# ============================================================================
# Request isolation
# ============================================================================


@pytest.mark.asyncio
async def test_service_does_not_mutate_original_request():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = make_request(
        frame_count=3,
        max_observations=2,
    )

    original_frame_count = len(request.frames)

    await service.analyze(request)

    assert len(request.frames) == original_frame_count


@pytest.mark.asyncio
async def test_service_bounds_provider_request_using_max_observations():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = make_request(
        frame_count=5,
        max_observations=2,
    )

    await service.analyze(request)

    assert len(provider.requests) == 1

    forwarded_request = provider.requests[0]

    assert len(forwarded_request.frames) == 2


@pytest.mark.asyncio
async def test_service_preserves_frame_order_when_bounding_request():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = make_request(
        frame_count=5,
        max_observations=3,
    )

    await service.analyze(request)

    forwarded_request = provider.requests[0]

    assert [
        frame.frame_index
        for frame in forwarded_request.frames
    ] == [0, 1, 2]


@pytest.mark.asyncio
async def test_service_preserves_frame_timestamps_when_bounding_request():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = make_request(
        frame_count=4,
        max_observations=2,
    )

    await service.analyze(request)

    forwarded_request = provider.requests[0]

    assert [
        frame.timestamp_seconds
        for frame in forwarded_request.frames
    ] == [1.0, 2.0]


# ============================================================================
# Result propagation
# ============================================================================


@pytest.mark.asyncio
async def test_service_returns_provider_vision_result():
    result = make_completed_result()

    provider = FakeVisionProvider(
        result=result,
    )

    service = VisionService(provider)

    returned = await service.analyze(
        make_request(),
    )

    assert returned is result


@pytest.mark.asyncio
async def test_service_preserves_provider_status():
    result = make_completed_result()

    provider = FakeVisionProvider(
        result=result,
    )

    service = VisionService(provider)

    returned = await service.analyze(
        make_request(),
    )

    assert returned.status == VisionStatus.COMPLETED


@pytest.mark.asyncio
async def test_service_preserves_provider_observations():
    result = make_completed_result()

    provider = FakeVisionProvider(
        result=result,
    )

    service = VisionService(provider)

    returned = await service.analyze(
        make_request(),
    )

    assert returned.observations == result.observations


@pytest.mark.asyncio
async def test_service_preserves_provider_metadata():
    result = make_completed_result()

    provider = FakeVisionProvider(
        result=result,
    )

    service = VisionService(provider)

    returned = await service.analyze(
        make_request(),
    )

    assert returned.metadata["provider"] == "fake-provider"

    assert (
        returned.metadata["custom_metadata"]
        == "preserved"
    )


# ============================================================================
# Service execution metadata
# ============================================================================


@pytest.mark.asyncio
async def test_service_adds_service_execution_metadata():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    result = await service.analyze(
        make_request(),
    )

    assert "service_execution" in result.metadata


@pytest.mark.asyncio
async def test_service_execution_metadata_contains_frame_counts():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    result = await service.analyze(
        make_request(
            frame_count=3,
        ),
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 3
    assert execution["processed_frames"] == 1
    assert execution["failed_frames"] == 2


@pytest.mark.asyncio
async def test_service_execution_metadata_contains_duration():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    result = await service.analyze(
        make_request(),
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert "duration_ms" in execution
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_does_not_overwrite_provider_metadata():
    result = make_completed_result(
        provider_name="custom-provider",
    )

    provider = FakeVisionProvider(
        result=result,
    )

    service = VisionService(provider)

    returned = await service.analyze(
        make_request(),
    )

    assert (
        returned.metadata["provider"]
        == "custom-provider"
    )


# ============================================================================
# Empty request
# ============================================================================


@pytest.mark.asyncio
async def test_empty_request_returns_no_frames():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = VisionRequest(
        frames=[],
    )

    result = await service.analyze(
        request,
    )

    assert result.status == VisionStatus.NO_FRAMES
    assert result.requested_frames == 0
    assert result.processed_frames == 0
    assert result.failed_frames == 0


@pytest.mark.asyncio
async def test_empty_request_does_not_invoke_provider():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    request = VisionRequest(
        frames=[],
    )

    await service.analyze(
        request,
    )

    assert provider.requests == []


# ============================================================================
# Provider failure propagation
# ============================================================================


@pytest.mark.asyncio
async def test_provider_exception_becomes_structured_failure():
    provider = FakeVisionProvider(
        exception=RuntimeError(
            "provider failed",
        ),
    )

    service = VisionService(provider)

    result = await service.analyze(
        make_request(
            frame_count=2,
        ),
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 2
    assert result.processed_frames == 0
    assert result.failed_frames == 2


@pytest.mark.asyncio
async def test_provider_exception_does_not_escape_service_boundary():
    provider = FakeVisionProvider(
        exception=RuntimeError(
            "provider failed",
        ),
    )

    service = VisionService(provider)

    result = await service.analyze(
        make_request(),
    )

    assert isinstance(
        result,
        VisionResult,
    )


@pytest.mark.asyncio
async def test_provider_exception_type_is_recorded():
    provider = FakeVisionProvider(
        exception=TimeoutError(
            "provider timeout",
        ),
    )

    service = VisionService(provider)

    result = await service.analyze(
        make_request(),
    )

    assert (
        result.metadata["exception_type"]
        == "TimeoutError"
    )


@pytest.mark.asyncio
async def test_provider_exception_message_is_recorded():
    provider = FakeVisionProvider(
        exception=RuntimeError(
            "provider failed",
        ),
    )

    service = VisionService(provider)

    result = await service.analyze(
        make_request(),
    )

    assert result.errors == [
        "provider failed",
    ]


# ============================================================================
# Invalid provider result
# ============================================================================


class InvalidResultProvider:
    """
    Deliberately violates the VisionProvider result contract.
    """

    name = "invalid-result-provider"

    def __init__(self) -> None:
        self.requests: list[VisionRequest] = []

    async def analyze(
        self,
        request: VisionRequest,
    ) -> object:
        self.requests.append(request)

        return {
            "status": "completed",
        }


@pytest.mark.asyncio
async def test_invalid_provider_result_becomes_structured_failure():
    provider = InvalidResultProvider()

    service = VisionService(provider)  # type: ignore[arg-type]

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_invalid_provider_result_type_is_recorded():
    provider = InvalidResultProvider()

    service = VisionService(provider)  # type: ignore[arg-type]

    result = await service.analyze(
        make_request(),
    )

    assert (
        result.metadata["result_type"]
        == "dict"
    )


# ============================================================================
# Local provider integration boundary
# ============================================================================


@pytest.mark.asyncio
async def test_local_provider_runs_through_vision_service():
    runtime = FakeLocalVisionRuntime()

    provider = LocalVisionProvider(
        runtime,
    )

    service = VisionService(
        provider,
    )

    result = await service.analyze(
        VisionRequest(
            frames=[
                make_frame(
                    frame_index=7,
                    timestamp_seconds=15.5,
                )
            ],
        ),
    )

    assert isinstance(
        result,
        VisionResult,
    )

    assert result.status == VisionStatus.COMPLETED


@pytest.mark.asyncio
async def test_local_provider_runtime_is_invoked_through_service():
    runtime = FakeLocalVisionRuntime()

    provider = LocalVisionProvider(
        runtime,
    )

    service = VisionService(
        provider,
    )

    await service.analyze(
        make_request(),
    )

    assert len(runtime.requests) == 1


@pytest.mark.asyncio
async def test_local_provider_result_metadata_reaches_service():
    runtime = FakeLocalVisionRuntime()

    provider = LocalVisionProvider(
        runtime,
    )

    service = VisionService(
        provider,
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.metadata["provider"] == "local"
    assert result.metadata["runtime"] == "fake-local-runtime"
    assert result.metadata["model"] == "fake-local-model"


@pytest.mark.asyncio
async def test_local_provider_runtime_metadata_is_preserved():
    runtime = FakeLocalVisionRuntime()

    provider = LocalVisionProvider(
        runtime,
    )

    service = VisionService(
        provider,
    )

    result = await service.analyze(
        make_request(),
    )

    assert (
        result.metadata["runtime_metadata"]["integration_test"]
        is True
    )


@pytest.mark.asyncio
async def test_local_provider_service_adds_execution_metadata():
    runtime = FakeLocalVisionRuntime()

    provider = LocalVisionProvider(
        runtime,
    )

    service = VisionService(
        provider,
    )

    result = await service.analyze(
        make_request(),
    )

    assert "service_execution" in result.metadata

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 1
    assert execution["failed_frames"] == 0