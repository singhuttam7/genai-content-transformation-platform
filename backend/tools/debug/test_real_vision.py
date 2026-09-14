from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Allow `app` imports when this file is executed directly.
BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ingestion.video.schemas import VideoFrame
from app.ingestion.video.vision import (
    VisionRequest,
    VisionService,
    VisionStatus,
)
from app.ingestion.video.vision_factory import create_vision_provider


FRAME_PATH = (
    BACKEND_ROOT
    / "test_data"
    / "video"
    / "frame_test"
    / "frame_001.jpg"
)


async def main() -> None:
    print("=" * 70)
    print("REAL APPLICATION-LEVEL VISION TEST")
    print("=" * 70)

    if not FRAME_PATH.exists():
        raise FileNotFoundError(
            f"Test frame not found: {FRAME_PATH}"
        )

    image_bytes = FRAME_PATH.read_bytes()

    print(f"\nFrame: {FRAME_PATH}")
    print(f"Frame size: {len(image_bytes):,} bytes")

    # ------------------------------------------------------------
    # Build the project's actual VideoFrame contract.
    # ------------------------------------------------------------
    frame = VideoFrame(
        timestamp_seconds=0.0,
        frame_index=0,
        image=image_bytes,
        metadata={
            "filename": FRAME_PATH.name,
            "mime_type": "image/jpeg",
            "source": "real-vision-debug",
        },
    )

    # ------------------------------------------------------------
    # Create the real production provider through the factory.
    # ------------------------------------------------------------
    provider = create_vision_provider()
    service = VisionService(provider)

    request = VisionRequest(
        frames=(frame,),
        prompt=(
            "Carefully inspect the image and describe what is visibly "
            "present. Identify the main objects, scene or setting, "
            "people if visible, readable text if present, and other "
            "important visual details. "
            "Do not describe anything that cannot be supported by "
            "the image."
        ),
        detail_level="standard",
        max_observations=5,
        metadata={
            "source_id": "real-vision-debug",
            "filename": FRAME_PATH.name,
            "mime_type": "image/jpeg",
        },
    )

    print("\nProvider:")
    print(f"  {type(provider).__name__}")

    print("\nVision request:")
    print(f"  Frames:        {len(request.frames)}")
    print(f"  Detail level:  {request.detail_level}")
    print(f"  Max outputs:   {request.max_observations}")

    print("\nCalling VisionService.analyze()...")
    print("This may take several seconds with Gemma 3 4B.\n")

    started = time.perf_counter()

    try:
        result = await service.analyze(request)
    finally:
        close = getattr(provider, "close", None)

        if close is not None:
            maybe_awaitable = close()

            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable

    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------
    # Display result.
    # ------------------------------------------------------------
    print("=" * 70)
    print("VISION RESULT")
    print("=" * 70)

    print(f"\nStatus: {result.status}")
    print(f"Duration: {elapsed:.2f} seconds")

    print("\nFrame counts:")
    print(f"  Requested: {result.requested_frames}")
    print(f"  Processed: {result.processed_frames}")
    print(f"  Failed:    {result.failed_frames}")

    print("\nObservations:")

    if not result.observations:
        print("  No observations returned.")

    for index, observation in enumerate(
        result.observations,
        start=1,
    ):
        print(f"\n  Observation {index}")
        print(f"    Frame index:     {observation.frame_index}")
        print(f"    Timestamp:       {observation.timestamp_seconds}")
        print(f"    Description:     {observation.description}")
        print(f"    Confidence:      {observation.confidence}")

        if observation.objects:
            print(f"    Objects:         {observation.objects}")

        if observation.entities:
            print(f"    Entities:        {observation.entities}")

        if observation.actions:
            print(f"    Actions:         {observation.actions}")

        if observation.scene:
            print(f"    Scene:           {observation.scene}")

        if observation.visible_text:
            print(f"    Visible text:    {observation.visible_text}")

    print("\nMetadata:")

    if result.metadata:
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  None")

    print("\nErrors:")

    if result.errors:
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("  None")

    print("\n" + "=" * 70)

    if (
        result.status == VisionStatus.COMPLETED
        and result.observations
    ):
        print("REAL VISION INTEGRATION: SUCCESS")
    elif result.status == VisionStatus.COMPLETED:
        print(
            "VISION REQUEST COMPLETED, "
            "BUT NO OBSERVATIONS WERE RETURNED"
        )
    else:
        print("REAL VISION INTEGRATION: FAILED")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())