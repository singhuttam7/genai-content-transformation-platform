import pytest

from app.ingestion.video.schemas import (
    VideoFrame,
    VisionObservation,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from app.ingestion.video.vision import VisionService


class FakeVisionProvider:
    """
    Deterministic provider used to test VisionService.
    """

    name = "fake-vision-provider"

    def __init__(
        self,
        *,
        result: VisionResult | None = None,
    ) -> None:
        self.calls = 0
        self.last_request: VisionRequest | None = None
        self._result = result

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        self.calls += 1
        self.last_request = request

        if self._result is not None:
            return self._result

        observations = [
            VisionObservation(
                timestamp_seconds=frame.timestamp_seconds,
                frame_index=frame.frame_index,
                description="Test visual observation.",
                confidence=0.95,
            )
            for frame in request.frames
        ]

        return VisionResult(
            status=VisionStatus.COMPLETED,
            observations=observations,
            requested_frames=len(request.frames),
            processed_frames=len(request.frames),
            failed_frames=0,
            metadata={
                "provider": self.name,
            },
        )


class FakeFailingVisionProvider:
    """
    Provider that raises an exception during analysis.
    """

    name = "failing-vision-provider"

    def __init__(
        self,
        exception: Exception,
    ) -> None:
        self.calls = 0
        self.last_request: VisionRequest | None = None
        self._exception = exception

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        self.calls += 1
        self.last_request = request

        raise self._exception


class FakeInvalidResultProvider:
    """
    Provider that violates the VisionProvider result contract.
    """

    name = "invalid-result-provider"

    def __init__(
        self,
        result: object,
    ) -> None:
        self.calls = 0
        self.last_request: VisionRequest | None = None
        self._result = result

    async def analyze(
        self,
        request: VisionRequest,
    ):
        self.calls += 1
        self.last_request = request

        return self._result


def make_frame(
    *,
    timestamp: float,
    frame_index: int,
) -> VideoFrame:
    return VideoFrame(
        timestamp_seconds=timestamp,
        frame_index=frame_index,
        image=b"\xff\xd8fake-jpeg-data\xff\xd9",
        width=1280,
        height=720,
    )


# ============================================================
# INITIALIZATION
# ============================================================


def test_service_requires_provider():
    with pytest.raises(ValueError, match="provider"):
        VisionService(None)


def test_service_exposes_configured_provider():
    provider = FakeVisionProvider()

    service = VisionService(provider)

    assert service.provider is provider


# ============================================================
# REQUEST VALIDATION
# ============================================================


@pytest.mark.asyncio
async def test_service_rejects_missing_request():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    with pytest.raises(ValueError, match="request"):
        await service.analyze(None)


# ============================================================
# EMPTY REQUEST
# ============================================================


@pytest.mark.asyncio
async def test_service_returns_no_frames_for_empty_request():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest()

    result = await service.analyze(request)

    assert result.status == VisionStatus.NO_FRAMES
    assert result.requested_frames == 0
    assert result.processed_frames == 0
    assert result.failed_frames == 0
    assert result.observations == []

    assert result.metadata["service"] == "vision"
    assert result.metadata["provider"] == provider.name

    assert result.metadata["service_execution"] == {
        "requested_frames": 0,
        "processed_frames": 0,
        "failed_frames": 0,
        "duration_ms": 0.0,
    }

    assert provider.calls == 0


# ============================================================
# PROVIDER INVOCATION
# ============================================================


@pytest.mark.asyncio
async def test_service_invokes_provider_exactly_once():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
        ]
    )

    await service.analyze(request)

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_service_passes_prepared_request_to_provider():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
        ],
        prompt="Describe the important visual content.",
        detail_level="detailed",
    )

    await service.analyze(request)

    assert provider.last_request is not None
    assert provider.last_request is not request

    assert provider.last_request.prompt == (
        "Describe the important visual content."
    )
    assert provider.last_request.detail_level == "detailed"


@pytest.mark.asyncio
async def test_service_forwards_bounded_request_to_provider():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
            make_frame(
                timestamp=4.0,
                frame_index=2,
            ),
        ],
        max_observations=2,
    )

    await service.analyze(request)

    assert provider.last_request is not None
    assert len(provider.last_request.frames) == 2

    assert [
        frame.frame_index
        for frame in provider.last_request.frames
    ] == [0, 1]


# ============================================================
# RESULT PRESERVATION
# ============================================================


@pytest.mark.asyncio
async def test_service_preserves_completed_result():
    expected = VisionResult(
        status=VisionStatus.COMPLETED,
        observations=[
            VisionObservation(
                timestamp_seconds=2.0,
                frame_index=1,
                description="Completed observation.",
                confidence=0.96,
            )
        ],
        requested_frames=1,
        processed_frames=1,
        failed_frames=0,
        metadata={
            "provider": "custom-provider",
            "model": "custom-model",
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=2.0,
                frame_index=1,
            )
        ]
    )

    result = await service.analyze(request)

    assert result is expected
    assert result.status == VisionStatus.COMPLETED
    assert result.observations[0].description == (
        "Completed observation."
    )

    assert result.metadata["provider"] == "custom-provider"
    assert result.metadata["model"] == "custom-model"

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 1
    assert execution["failed_frames"] == 0
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_preserves_partial_result():
    expected = VisionResult(
        status=VisionStatus.PARTIAL,
        observations=[
            VisionObservation(
                timestamp_seconds=0.0,
                frame_index=0,
                description="First frame processed.",
                confidence=0.91,
            ),
            VisionObservation(
                timestamp_seconds=4.0,
                frame_index=2,
                description="Third frame processed.",
                confidence=0.88,
            ),
        ],
        requested_frames=3,
        processed_frames=2,
        failed_frames=1,
        errors=[
            "Frame 1 failed.",
        ],
        metadata={
            "provider": "partial-provider",
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
            make_frame(
                timestamp=4.0,
                frame_index=2,
            ),
        ]
    )

    result = await service.analyze(request)

    assert result is expected
    assert result.status == VisionStatus.PARTIAL
    assert result.requested_frames == 3
    assert result.processed_frames == 2
    assert result.failed_frames == 1
    assert len(result.observations) == 2
    assert result.errors == [
        "Frame 1 failed.",
    ]

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 3
    assert execution["processed_frames"] == 2
    assert execution["failed_frames"] == 1
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_preserves_failed_result():
    expected = VisionResult(
        status=VisionStatus.FAILED,
        requested_frames=2,
        processed_frames=0,
        failed_frames=2,
        errors=[
            "Vision provider failed.",
        ],
        metadata={
            "provider": "failing-provider",
            "retryable": False,
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
        ]
    )

    result = await service.analyze(request)

    assert result is expected
    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 2
    assert result.processed_frames == 0
    assert result.failed_frames == 2
    assert result.errors == [
        "Vision provider failed.",
    ]

    assert result.metadata["retryable"] is False

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 2
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 2
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_preserves_provider_unavailable_result():
    expected = VisionResult(
        status=VisionStatus.PROVIDER_UNAVAILABLE,
        requested_frames=1,
        processed_frames=0,
        failed_frames=1,
        errors=[
            "Vision provider unavailable.",
        ],
        metadata={
            "provider": "unavailable-provider",
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=3.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result is expected
    assert result.status == VisionStatus.PROVIDER_UNAVAILABLE
    assert result.errors == [
        "Vision provider unavailable.",
    ]

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 1
    assert execution["duration_ms"] >= 0.0


# ============================================================
# PROVIDER EXCEPTION HANDLING
# ============================================================


@pytest.mark.asyncio
async def test_service_converts_provider_exception_to_failed_result():
    provider = FakeFailingVisionProvider(
        RuntimeError(
            "Vision model execution failed."
        )
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
        ]
    )

    result = await service.analyze(request)

    assert provider.calls == 1

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 2
    assert result.processed_frames == 0
    assert result.failed_frames == 2

    assert result.errors == [
        "Vision model execution failed.",
    ]

    assert result.metadata["service"] == "vision"
    assert result.metadata["provider"] == (
        "failing-vision-provider"
    )
    assert result.metadata["exception_type"] == (
        "RuntimeError"
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 2
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 2
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_converts_timeout_to_failed_result():
    provider = FakeFailingVisionProvider(
        TimeoutError(
            "Vision inference timed out."
        )
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=5.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors == [
        "Vision inference timed out.",
    ]

    assert result.metadata["exception_type"] == (
        "TimeoutError"
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 1
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_handles_exception_without_message():
    provider = FakeFailingVisionProvider(
        RuntimeError()
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=1.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors == [
        "Vision provider execution failed.",
    ]

    assert result.metadata["exception_type"] == (
        "RuntimeError"
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 1
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_provider_exception_does_not_expose_exception_object():
    provider = FakeFailingVisionProvider(
        ValueError(
            "Invalid model configuration."
        )
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result.status == VisionStatus.FAILED

    assert result.errors == [
        "Invalid model configuration.",
    ]

    assert result.metadata["exception_type"] == (
        "ValueError"
    )

    assert all(
        not isinstance(error, Exception)
        for error in result.errors
    )


# ============================================================
# RESULT CONTRACT ENFORCEMENT
# ============================================================


@pytest.mark.asyncio
async def test_service_accepts_valid_vision_result():
    expected = VisionResult(
        status=VisionStatus.COMPLETED,
        observations=[
            VisionObservation(
                timestamp_seconds=1.0,
                frame_index=0,
                description="Valid provider result.",
                confidence=0.94,
            )
        ],
        requested_frames=1,
        processed_frames=1,
        failed_frames=0,
        metadata={
            "provider": "valid-provider",
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=1.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result is expected
    assert isinstance(result, VisionResult)
    assert result.status == VisionStatus.COMPLETED

    assert result.metadata["provider"] == (
        "valid-provider"
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 1
    assert execution["failed_frames"] == 0
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_rejects_none_provider_result():
    provider = FakeInvalidResultProvider(
        result=None,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
        ]
    )

    result = await service.analyze(request)

    assert provider.calls == 1

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 2
    assert result.processed_frames == 0
    assert result.failed_frames == 2

    assert result.errors == [
        "Vision provider returned an invalid result type."
    ]

    assert result.metadata["service"] == "vision"
    assert result.metadata["provider"] == (
        "invalid-result-provider"
    )
    assert result.metadata["result_type"] == "NoneType"

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 2
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 2
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_rejects_dictionary_provider_result():
    provider = FakeInvalidResultProvider(
        result={
            "status": "completed",
            "observations": [],
        },
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=3.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors == [
        "Vision provider returned an invalid result type."
    ]

    assert result.metadata["result_type"] == "dict"

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 1
    assert execution["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_service_does_not_expose_invalid_provider_object():
    invalid_result = object()

    provider = FakeInvalidResultProvider(
        result=invalid_result,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=4.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    assert result.status == VisionStatus.FAILED

    assert result.errors == [
        "Vision provider returned an invalid result type."
    ]

    assert result.metadata["result_type"] == "object"

    assert invalid_result not in result.metadata.values()

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 1
    assert execution["duration_ms"] >= 0.0


# ============================================================
# PROVIDER METADATA PRESERVATION
# ============================================================


@pytest.mark.asyncio
async def test_service_preserves_provider_metadata():
    expected = VisionResult(
        status=VisionStatus.COMPLETED,
        requested_frames=2,
        processed_frames=2,
        failed_frames=0,
        metadata={
            "provider": "local-provider",
            "model": "vision-model",
            "device": "cpu",
            "custom_metric": 0.87,
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
        ]
    )

    result = await service.analyze(request)

    assert result.metadata["provider"] == (
        "local-provider"
    )
    assert result.metadata["model"] == (
        "vision-model"
    )
    assert result.metadata["device"] == "cpu"
    assert result.metadata["custom_metric"] == 0.87

    assert "service_execution" in result.metadata


@pytest.mark.asyncio
async def test_service_execution_metadata_matches_result_counts():
    expected = VisionResult(
        status=VisionStatus.PARTIAL,
        requested_frames=5,
        processed_frames=3,
        failed_frames=2,
        metadata={
            "provider": "count-provider",
        },
    )

    provider = FakeVisionProvider(
        result=expected,
    )

    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            )
        ]
    )

    result = await service.analyze(request)

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == (
        result.requested_frames
    )
    assert execution["processed_frames"] == (
        result.processed_frames
    )
    assert execution["failed_frames"] == (
        result.failed_frames
    )


# ============================================================
# TIMESTAMP / ORDER PRESERVATION
# ============================================================


@pytest.mark.asyncio
async def test_service_preserves_frame_timestamps():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=1.5,
                frame_index=0,
            ),
            make_frame(
                timestamp=3.5,
                frame_index=1,
            ),
            make_frame(
                timestamp=7.0,
                frame_index=2,
            ),
        ],
    )

    result = await service.analyze(request)

    timestamps = [
        observation.timestamp_seconds
        for observation in result.observations
    ]

    assert timestamps == [
        1.5,
        3.5,
        7.0,
    ]


@pytest.mark.asyncio
async def test_service_preserves_frame_order():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            ),
            make_frame(
                timestamp=2.0,
                frame_index=1,
            ),
            make_frame(
                timestamp=4.0,
                frame_index=2,
            ),
        ],
    )

    result = await service.analyze(request)

    frame_indices = [
        observation.frame_index
        for observation in result.observations
    ]

    assert frame_indices == [0, 1, 2]


# ============================================================
# REQUEST IMMUTABILITY
# ============================================================


@pytest.mark.asyncio
async def test_service_does_not_modify_original_request():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    frames = [
        make_frame(
            timestamp=2.0,
            frame_index=1,
        ),
        make_frame(
            timestamp=4.0,
            frame_index=2,
        ),
        make_frame(
            timestamp=6.0,
            frame_index=3,
        ),
    ]

    request = VisionRequest(
        frames=frames,
        prompt="Describe this video.",
        detail_level="detailed",
        max_observations=2,
        metadata={
            "source": "video",
        },
    )

    original_frame_count = len(request.frames)
    original_prompt = request.prompt
    original_detail_level = request.detail_level
    original_metadata = request.metadata.copy()

    await service.analyze(request)

    assert len(request.frames) == original_frame_count
    assert request.prompt == original_prompt
    assert request.detail_level == original_detail_level
    assert request.metadata == original_metadata


@pytest.mark.asyncio
async def test_service_provider_request_is_independent_copy():
    provider = FakeVisionProvider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=2.0,
                frame_index=0,
            )
        ],
        metadata={
            "source": "video",
        },
    )

    await service.analyze(request)

    assert provider.last_request is not None
    assert provider.last_request is not request
    assert provider.last_request.metadata == request.metadata