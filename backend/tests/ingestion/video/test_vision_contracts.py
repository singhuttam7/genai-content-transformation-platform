import pytest
from pydantic import ValidationError

from app.ingestion.video.schemas import (
    VideoFrame,
    VisionObservation,
    VisionRequest,
    VisionResult,
    VisionStatus,
)


def make_frame(
    *,
    timestamp: float = 2.0,
    frame_index: int = 1,
) -> VideoFrame:
    return VideoFrame(
        timestamp_seconds=timestamp,
        frame_index=frame_index,
        image=b"\xff\xd8fake-jpeg-data\xff\xd9",
        width=1280,
        height=720,
    )


# ============================================================
# VISION STATUS
# ============================================================


def test_vision_status_values():
    assert VisionStatus.NOT_REQUESTED.value == "not_requested"
    assert VisionStatus.PROCESSING.value == "processing"
    assert VisionStatus.COMPLETED.value == "completed"
    assert VisionStatus.PARTIAL.value == "partial"
    assert VisionStatus.NO_FRAMES.value == "no_frames"
    assert VisionStatus.PROVIDER_UNAVAILABLE.value == "provider_unavailable"
    assert VisionStatus.FAILED.value == "failed"


# ============================================================
# VISION REQUEST
# ============================================================


def test_vision_request_defaults():
    request = VisionRequest()

    assert request.frames == []
    assert request.prompt is None
    assert request.detail_level == "standard"
    assert request.max_observations is None
    assert request.metadata == {}


def test_vision_request_accepts_frames():
    frame = make_frame(timestamp=4.0, frame_index=2)

    request = VisionRequest(
        frames=[frame],
        prompt="Describe the scene.",
        detail_level="detailed",
        max_observations=10,
    )

    assert len(request.frames) == 1
    assert request.frames[0].timestamp_seconds == 4.0
    assert request.frames[0].frame_index == 2
    assert request.prompt == "Describe the scene."
    assert request.detail_level == "detailed"
    assert request.max_observations == 10


def test_vision_request_rejects_invalid_max_observations():
    with pytest.raises(ValidationError):
        VisionRequest(max_observations=0)


def test_vision_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VisionRequest(unknown_field="value")


# ============================================================
# VISION OBSERVATION
# ============================================================


def test_vision_observation_defaults():
    observation = VisionObservation(
        timestamp_seconds=2.0,
        frame_index=1,
    )

    assert observation.timestamp_seconds == 2.0
    assert observation.frame_index == 1
    assert observation.description is None
    assert observation.objects == []
    assert observation.entities == []
    assert observation.actions == []
    assert observation.scene is None
    assert observation.visible_text is None
    assert observation.confidence is None
    assert observation.metadata == {}


def test_vision_observation_accepts_structured_content():
    observation = VisionObservation(
        timestamp_seconds=4.0,
        frame_index=2,
        description="A person is standing near a workstation.",
        objects=["person", "computer", "desk"],
        entities=["workstation"],
        actions=["standing"],
        scene="office",
        visible_text="Security Operations Center",
        confidence=0.93,
    )

    assert observation.timestamp_seconds == 4.0
    assert observation.frame_index == 2
    assert observation.description == (
        "A person is standing near a workstation."
    )
    assert observation.objects == [
        "person",
        "computer",
        "desk",
    ]
    assert observation.entities == ["workstation"]
    assert observation.actions == ["standing"]
    assert observation.scene == "office"
    assert observation.visible_text == "Security Operations Center"
    assert observation.confidence == 0.93


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_vision_observation_rejects_invalid_confidence(
    confidence: float,
):
    with pytest.raises(ValidationError):
        VisionObservation(
            timestamp_seconds=1.0,
            frame_index=0,
            confidence=confidence,
        )


def test_vision_observation_rejects_negative_timestamp():
    with pytest.raises(ValidationError):
        VisionObservation(
            timestamp_seconds=-1.0,
            frame_index=0,
        )


def test_vision_observation_rejects_negative_frame_index():
    with pytest.raises(ValidationError):
        VisionObservation(
            timestamp_seconds=1.0,
            frame_index=-1,
        )


def test_vision_observation_preserves_metadata():
    observation = VisionObservation(
        timestamp_seconds=6.0,
        frame_index=3,
        metadata={
            "provider": "test-provider",
            "model": "test-model",
        },
    )

    assert observation.metadata["provider"] == "test-provider"
    assert observation.metadata["model"] == "test-model"


def test_vision_observation_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VisionObservation(
            timestamp_seconds=1.0,
            frame_index=0,
            unknown_field="invalid",
        )


# ============================================================
# VISION RESULT
# ============================================================


def test_vision_result_defaults():
    result = VisionResult(
        status=VisionStatus.NOT_REQUESTED,
    )

    assert result.status == VisionStatus.NOT_REQUESTED
    assert result.observations == []
    assert result.requested_frames == 0
    assert result.processed_frames == 0
    assert result.failed_frames == 0
    assert result.errors == []
    assert result.metadata == {}


def test_vision_result_accepts_completed_observations():
    observations = [
        VisionObservation(
            timestamp_seconds=0.0,
            frame_index=0,
            description="Opening scene.",
            confidence=0.90,
        ),
        VisionObservation(
            timestamp_seconds=2.0,
            frame_index=1,
            description="A person enters the scene.",
            confidence=0.94,
        ),
    ]

    result = VisionResult(
        status=VisionStatus.COMPLETED,
        observations=observations,
        requested_frames=2,
        processed_frames=2,
        failed_frames=0,
    )

    assert result.status == VisionStatus.COMPLETED
    assert len(result.observations) == 2
    assert result.requested_frames == 2
    assert result.processed_frames == 2
    assert result.failed_frames == 0


def test_vision_result_supports_partial_results():
    observation = VisionObservation(
        timestamp_seconds=2.0,
        frame_index=1,
        description="Successfully analyzed frame.",
        confidence=0.91,
    )

    result = VisionResult(
        status=VisionStatus.PARTIAL,
        observations=[observation],
        requested_frames=3,
        processed_frames=1,
        failed_frames=2,
        errors=[
            "Frame 0 failed.",
            "Frame 2 failed.",
        ],
    )

    assert result.status == VisionStatus.PARTIAL
    assert len(result.observations) == 1
    assert result.requested_frames == 3
    assert result.processed_frames == 1
    assert result.failed_frames == 2
    assert len(result.errors) == 2


def test_vision_result_supports_no_frames():
    result = VisionResult(
        status=VisionStatus.NO_FRAMES,
    )

    assert result.status == VisionStatus.NO_FRAMES
    assert result.observations == []
    assert result.requested_frames == 0
    assert result.processed_frames == 0
    assert result.failed_frames == 0


def test_vision_result_supports_provider_unavailable():
    result = VisionResult(
        status=VisionStatus.PROVIDER_UNAVAILABLE,
        errors=["Vision provider is unavailable."],
    )

    assert result.status == VisionStatus.PROVIDER_UNAVAILABLE
    assert result.errors == [
        "Vision provider is unavailable."
    ]


def test_vision_result_rejects_negative_counts():
    with pytest.raises(ValidationError):
        VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=-1,
        )

    with pytest.raises(ValidationError):
        VisionResult(
            status=VisionStatus.FAILED,
            processed_frames=-1,
        )

    with pytest.raises(ValidationError):
        VisionResult(
            status=VisionStatus.FAILED,
            failed_frames=-1,
        )


def test_vision_result_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VisionResult(
            status=VisionStatus.FAILED,
            unknown_field="invalid",
        )


# ============================================================
# TIMESTAMP PRESERVATION
# ============================================================


def test_vision_observation_preserves_original_frame_timestamp():
    frame = make_frame(
        timestamp=8.0,
        frame_index=4,
    )

    observation = VisionObservation(
        timestamp_seconds=frame.timestamp_seconds,
        frame_index=frame.frame_index,
        description="Test observation.",
    )

    assert observation.timestamp_seconds == frame.timestamp_seconds
    assert observation.frame_index == frame.frame_index