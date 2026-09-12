import pytest

from app.ingestion.video.result_aggregation import (
    CapabilityResultStatus,
    aggregate_video_status,
)
from app.ingestion.video.schemas import VideoProcessingStatus


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (
            [],
            VideoProcessingStatus.NOT_REQUESTED,
        ),
        (
            [CapabilityResultStatus.SUCCESS],
            VideoProcessingStatus.COMPLETED,
        ),
        (
            [
                CapabilityResultStatus.SUCCESS,
                CapabilityResultStatus.SUCCESS,
            ],
            VideoProcessingStatus.COMPLETED,
        ),
        (
            [
                CapabilityResultStatus.SUCCESS,
                CapabilityResultStatus.FAILED,
            ],
            VideoProcessingStatus.PARTIAL,
        ),
        (
            [
                CapabilityResultStatus.FAILED,
                CapabilityResultStatus.SUCCESS,
                CapabilityResultStatus.UNAVAILABLE,
            ],
            VideoProcessingStatus.PARTIAL,
        ),
        (
            [
                CapabilityResultStatus.FAILED,
            ],
            VideoProcessingStatus.FAILED,
        ),
        (
            [
                CapabilityResultStatus.UNAVAILABLE,
            ],
            VideoProcessingStatus.FAILED,
        ),
        (
            [
                CapabilityResultStatus.FAILED,
                CapabilityResultStatus.UNAVAILABLE,
            ],
            VideoProcessingStatus.FAILED,
        ),
        (
            [
                CapabilityResultStatus.PROCESSING,
                CapabilityResultStatus.SUCCESS,
            ],
            VideoProcessingStatus.PROCESSING,
        ),
        (
            [
                CapabilityResultStatus.PROCESSING,
                CapabilityResultStatus.FAILED,
            ],
            VideoProcessingStatus.PROCESSING,
        ),
    ],
)
def test_aggregate_video_status(statuses, expected):
    assert aggregate_video_status(statuses) == expected


def test_partial_result_when_one_capability_succeeds():
    status = aggregate_video_status(
        [
            CapabilityResultStatus.SUCCESS,
            CapabilityResultStatus.FAILED,
        ]
    )

    assert status == VideoProcessingStatus.PARTIAL


def test_all_capabilities_successful():
    status = aggregate_video_status(
        [
            CapabilityResultStatus.SUCCESS,
            CapabilityResultStatus.SUCCESS,
            CapabilityResultStatus.SUCCESS,
        ]
    )

    assert status == VideoProcessingStatus.COMPLETED


def test_all_capabilities_failed():
    status = aggregate_video_status(
        [
            CapabilityResultStatus.FAILED,
            CapabilityResultStatus.FAILED,
        ]
    )

    assert status == VideoProcessingStatus.FAILED