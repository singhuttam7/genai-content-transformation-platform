import pytest

from app.ingestion.video.provider import VisionProvider
from app.ingestion.video.schemas import (
    VideoFrame,
    VisionRequest,
    VisionResult,
    VisionStatus,
)


class FakeVisionProvider:
    """
    Minimal provider implementation used to verify the
    VisionProvider protocol contract.
    """

    name = "fake-vision-provider"

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        observations = []

        for frame in request.frames:
            observations.append(
                {
                    "timestamp_seconds": frame.timestamp_seconds,
                    "frame_index": frame.frame_index,
                    "description": "Test visual observation.",
                }
            )

        return VisionResult(
            status=VisionStatus.COMPLETED,
            observations=observations,
            requested_frames=len(request.frames),
            processed_frames=len(request.frames),
            failed_frames=0,
        )


class FakeFailingVisionProvider:
    """
    Provider used to verify that provider failures can be
    represented through the provider-independent result.
    """

    name = "fake-failing-vision-provider"

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        return VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=len(request.frames),
            processed_frames=0,
            failed_frames=len(request.frames),
            errors=["Vision analysis failed."],
        )


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
# PROVIDER CONTRACT
# ============================================================


@pytest.mark.asyncio
async def test_fake_provider_implements_vision_provider_protocol():
    provider = FakeVisionProvider()

    assert provider.name == "fake-vision-provider"
    assert hasattr(provider, "analyze")

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=0.0,
                frame_index=0,
            )
        ]
    )

    result = await provider.analyze(request)

    assert isinstance(result, VisionResult)
    assert result.status == VisionStatus.COMPLETED


@pytest.mark.asyncio
async def test_provider_accepts_provider_independent_vision_request():
    provider = FakeVisionProvider()

    frames = [
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

    request = VisionRequest(
        frames=frames,
        prompt="Describe the important visual content.",
        detail_level="detailed",
        metadata={
            "source": "video",
        },
    )

    result = await provider.analyze(request)

    assert result.status == VisionStatus.COMPLETED
    assert result.requested_frames == 3
    assert result.processed_frames == 3
    assert result.failed_frames == 0
    assert len(result.observations) == 3


@pytest.mark.asyncio
async def test_provider_preserves_frame_timestamps():
    provider = FakeVisionProvider()

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
        ]
    )

    result = await provider.analyze(request)

    timestamps = [
        observation.timestamp_seconds
        for observation in result.observations
    ]

    frame_indices = [
        observation.frame_index
        for observation in result.observations
    ]

    assert timestamps == [1.5, 3.5, 7.0]
    assert frame_indices == [0, 1, 2]


@pytest.mark.asyncio
async def test_provider_preserves_frame_order():
    provider = FakeVisionProvider()

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

    result = await provider.analyze(request)

    assert [
        observation.frame_index
        for observation in result.observations
    ] == [0, 1, 2]


@pytest.mark.asyncio
async def test_provider_supports_empty_frame_request():
    provider = FakeVisionProvider()

    request = VisionRequest()

    result = await provider.analyze(request)

    assert isinstance(result, VisionResult)
    assert result.status == VisionStatus.COMPLETED
    assert result.requested_frames == 0
    assert result.processed_frames == 0
    assert result.failed_frames == 0
    assert result.observations == []


@pytest.mark.asyncio
async def test_provider_can_return_failure_result():
    provider = FakeFailingVisionProvider()

    request = VisionRequest(
        frames=[
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

    result = await provider.analyze(request)

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 2
    assert result.processed_frames == 0
    assert result.failed_frames == 2
    assert result.errors == [
        "Vision analysis failed.",
    ]


@pytest.mark.asyncio
async def test_provider_identity_is_exposed():
    provider = FakeVisionProvider()

    assert isinstance(provider.name, str)
    assert provider.name


@pytest.mark.asyncio
async def test_provider_returns_provider_independent_result():
    provider = FakeVisionProvider()

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=6.0,
                frame_index=3,
            )
        ]
    )

    result = await provider.analyze(request)

    assert isinstance(result, VisionResult)

    observation = result.observations[0]

    assert observation.timestamp_seconds == 6.0
    assert observation.frame_index == 3
    assert observation.description == "Test visual observation."


@pytest.mark.asyncio
async def test_provider_does_not_modify_input_frames():
    provider = FakeVisionProvider()

    frame = make_frame(
        timestamp=5.0,
        frame_index=2,
    )

    request = VisionRequest(
        frames=[frame],
    )

    original_timestamp = frame.timestamp_seconds
    original_frame_index = frame.frame_index
    original_image = frame.image

    await provider.analyze(request)

    assert frame.timestamp_seconds == original_timestamp
    assert frame.frame_index == original_frame_index
    assert frame.image == original_image


@pytest.mark.asyncio
async def test_provider_receives_metadata_without_provider_specific_contract():
    provider = FakeVisionProvider()

    request = VisionRequest(
        frames=[
            make_frame(
                timestamp=2.0,
                frame_index=1,
            )
        ],
        metadata={
            "source_id": "test-source",
            "filename": "sample.mp4",
            "language": "en",
        },
    )

    result = await provider.analyze(request)

    assert result.status == VisionStatus.COMPLETED
    assert result.requested_frames == 1