from __future__ import annotations

from enum import StrEnum

from app.ingestion.video.schemas import VideoProcessingStatus


class CapabilityResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    PROCESSING = "processing"


def aggregate_video_status(
    capability_statuses: list[CapabilityResultStatus],
) -> VideoProcessingStatus:
    """
    Determine the overall video-processing state from
    individual capability states.

    This function intentionally knows nothing about
    ASR, vision, OCR, or any concrete provider.
    """

    if not capability_statuses:
        return VideoProcessingStatus.NOT_REQUESTED

    if any(
        status == CapabilityResultStatus.PROCESSING
        for status in capability_statuses
    ):
        return VideoProcessingStatus.PROCESSING

    successful = sum(
        status == CapabilityResultStatus.SUCCESS
        for status in capability_statuses
    )

    failed = sum(
        status == CapabilityResultStatus.FAILED
        for status in capability_statuses
    )

    unavailable = sum(
        status == CapabilityResultStatus.UNAVAILABLE
        for status in capability_statuses
    )

    total = len(capability_statuses)

    if successful == total:
        return VideoProcessingStatus.COMPLETED

    if successful > 0:
        return VideoProcessingStatus.PARTIAL

    if failed + unavailable == total:
        return VideoProcessingStatus.FAILED

    return VideoProcessingStatus.PARTIAL