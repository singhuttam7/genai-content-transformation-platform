from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ingestion.video.schemas import VideoFrame, VisionRequest
from app.ingestion.video.providers.local.prompting import (
    DefaultVisionPromptBuilder,
)
from app.ingestion.video.providers.local.serialization import (
    Base64FrameSerializer,
)

FRAME_PATH = (
    BACKEND_ROOT
    / "test_data"
    / "video"
    / "frame_test"
    / "frame_001.jpg"
)


def main() -> None:
    image_bytes = FRAME_PATH.read_bytes()

    frame = VideoFrame(
        timestamp_seconds=0.0,
        frame_index=0,
        image=image_bytes,
        metadata={
            "filename": FRAME_PATH.name,
            "mime_type": "image/jpeg",
        },
    )

    request = VisionRequest(
        frames=(frame,),
        prompt=(
            "Carefully inspect the image and describe what is visibly "
            "present. Identify the main objects, scene or setting, "
            "people if visible, readable text if present, and other "
            "important visual details. Do not describe anything that "
            "cannot be supported by the image."
        ),
        detail_level="standard",
        max_observations=5,
        metadata={
            "source_id": "real-vision-debug",
            "filename": FRAME_PATH.name,
            "mime_type": "image/jpeg",
        },
    )

    prompt = DefaultVisionPromptBuilder().build(request)

    serialized = Base64FrameSerializer().serialize(frame)

    payload = {
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "system",
                "content": prompt.system_prompt,
            },
            {
                "role": "user",
                "content": prompt.user_prompt,
                "images": [serialized.data],
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
        "keep_alive": "5m",
    }

    print("=" * 70)
    print("EXACT APPLICATION PAYLOAD ? OLLAMA")
    print("=" * 70)

    print(f"\nImage bytes: {len(image_bytes):,}")
    print(f"Base64 length: {len(serialized.data):,}")
    print(f"System prompt length: {len(prompt.system_prompt):,}")
    print(f"User prompt length: {len(prompt.user_prompt):,}")

    print("\nUser prompt:")
    print("-" * 70)
    print(prompt.user_prompt)
    print("-" * 70)

    print("\nSending exact application-generated request...")

    started = time.perf_counter()

    with httpx.Client(
        base_url="http://localhost:11434",
        timeout=150.0,
    ) as client:
        response = client.post(
            "/api/chat",
            json=payload,
        )

    elapsed = time.perf_counter() - started

    print(f"\nHTTP status: {response.status_code}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    response.raise_for_status()

    data = response.json()

    print("\nOllama response:")
    print(json.dumps(data, indent=2))

    message = data.get("message", {})

    print("\n" + "=" * 70)
    print("MODEL CONTENT")
    print("=" * 70)
    print(message.get("content", ""))

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
