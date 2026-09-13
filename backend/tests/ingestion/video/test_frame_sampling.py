from __future__ import annotations

import pytest

from app.ingestion.video.sampling import (
    FrameSamplingPolicy,
)
from app.ingestion.video.schemas import (
    FrameExtractionRequest,
)


@pytest.fixture
def policy() -> FrameSamplingPolicy:
    return FrameSamplingPolicy()


def test_default_sampling(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest()

    timestamps = policy.generate_timestamps(
        duration_seconds=8.15,
        request=request,
    )

    assert timestamps == [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]


def test_sampling_with_custom_interval(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        interval_seconds=1.5,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=5.0,
        request=request,
    )

    assert timestamps == [
        0.0,
        1.5,
        3.0,
        4.5,
    ]


def test_sampling_with_start_time(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        interval_seconds=2.0,
        start_time_seconds=2.0,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=8.0,
        request=request,
    )

    assert timestamps == [
        2.0,
        4.0,
        6.0,
        8.0,
    ]


def test_sampling_with_end_time(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        interval_seconds=2.0,
        start_time_seconds=2.0,
        end_time_seconds=7.0,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=10.0,
        request=request,
    )

    assert timestamps == [
        2.0,
        4.0,
        6.0,
    ]


def test_end_time_is_clamped_to_video_duration(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        interval_seconds=2.0,
        end_time_seconds=20.0,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=5.0,
        request=request,
    )

    assert timestamps == [
        0.0,
        2.0,
        4.0,
    ]


def test_max_frames_limits_output(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        interval_seconds=1.0,
        max_frames=3,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=10.0,
        request=request,
    )

    assert timestamps == [
        0.0,
        1.0,
        2.0,
    ]


def test_start_after_duration_returns_empty(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        start_time_seconds=10.0,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=8.0,
        request=request,
    )

    assert timestamps == []


def test_end_before_start_returns_empty(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        start_time_seconds=6.0,
        end_time_seconds=4.0,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=10.0,
        request=request,
    )

    assert timestamps == []


def test_zero_duration_video_returns_empty(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest()

    timestamps = policy.generate_timestamps(
        duration_seconds=0.0,
        request=request,
    )

    assert timestamps == []


def test_timestamps_are_ordered(
    policy: FrameSamplingPolicy,
) -> None:
    request = FrameExtractionRequest(
        interval_seconds=0.75,
    )

    timestamps = policy.generate_timestamps(
        duration_seconds=5.0,
        request=request,
    )

    assert timestamps == sorted(
        timestamps
    )

    assert len(timestamps) == len(
        set(timestamps)
    )