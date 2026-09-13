from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.ingestion.video.providers.local.config import (  # noqa: E402
    LocalVisionConfig,
)
from app.ingestion.video.providers.local.ollama import (  # noqa: E402
    OllamaVisionRuntime,
)
from app.ingestion.video.providers.local.runtime import (  # noqa: E402
    VisionRuntimeRequest,
)
from app.ingestion.video.providers.local.serialization import (  # noqa: E402
    SerializedFrame,
)


# ---------------------------------------------------------------------------
# Benchmark frame
#
# Use one small frame for the initial performance test.
# ---------------------------------------------------------------------------

FRAME_PATH = (
    BACKEND_ROOT
    / "test_data"
    / "vision"
    / "real_test.jpeg"
)


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemma3:4b"

OLLAMA_BASE_URL = "http://localhost:11434"

# Keep this high enough for the benchmark.
# We are measuring actual inference latency, not enforcing
# production timeout behavior yet.
TIMEOUT_SECONDS = 300.0

TEMPERATURE = 0.1

KEEP_ALIVE = "5m"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # -----------------------------------------------------------------------
    # Validate frame
    # -----------------------------------------------------------------------

    if not FRAME_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark frame not found: {FRAME_PATH}"
        )

    frame_bytes = FRAME_PATH.read_bytes()

    if not frame_bytes:
        raise ValueError(
            "Benchmark frame is empty."
        )

    # -----------------------------------------------------------------------
    # Serialize frame
    # -----------------------------------------------------------------------

    encoded_frame = base64.b64encode(
        frame_bytes
    ).decode("ascii")

    serialized_frame = SerializedFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        mime_type="image/jpeg",
        data=encoded_frame,
    )

    # -----------------------------------------------------------------------
    # Runtime request
    # -----------------------------------------------------------------------

    request = VisionRuntimeRequest(
        model=MODEL_NAME,

        system_prompt=(
            "You are a visual analysis component in a "
            "multimodal content transformation platform. "
            "Analyze the supplied image carefully. "
            "Describe only information that is visually "
            "supported. Do not invent facts."
        ),

        # Intentionally short prompt for benchmarking.
        #
        # We want to determine whether the bottleneck is
        # the vision inference itself rather than a large
        # prompt.
        user_prompt=(
            "Briefly describe what is visible in this image."
        ),

        frames=(serialized_frame,),

        temperature=TEMPERATURE,

        keep_alive=KEEP_ALIVE,
    )

    # -----------------------------------------------------------------------
    # Ollama configuration
    # -----------------------------------------------------------------------

    config = LocalVisionConfig(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=TIMEOUT_SECONDS,
        max_retries=0,
        temperature=TEMPERATURE,
        keep_alive=KEEP_ALIVE,
    )

    runtime = OllamaVisionRuntime(config)

    try:
        # -------------------------------------------------------------------
        # Header
        # -------------------------------------------------------------------

        print("=" * 70)
        print("OLLAMA VISION FRAME BENCHMARK")
        print("=" * 70)

        print(f"Frame:       {FRAME_PATH}")
        print(f"Frame size:  {len(frame_bytes):,} bytes")
        print(f"Model:       {MODEL_NAME}")
        print(f"Runtime:     {runtime.name}")
        print(f"Endpoint:    {OLLAMA_BASE_URL}/api/chat")
        print(f"Timeout:     {TIMEOUT_SECONDS:.0f} seconds")

        print()
        print("Benchmark prompt:")
        print(request.user_prompt)

        print()
        print("-" * 70)
        print("Checking Ollama vision inference...")
        print("-" * 70)
        print()

        # -------------------------------------------------------------------
        # Measure inference time
        # -------------------------------------------------------------------

        started_at = time.perf_counter()

        response = await runtime.generate(request)

        elapsed_seconds = (
            time.perf_counter() - started_at
        )

        # -------------------------------------------------------------------
        # Response
        # -------------------------------------------------------------------

        print("=" * 70)
        print("VISION RESPONSE")
        print("=" * 70)

        print(response.text)

        print()

        # -------------------------------------------------------------------
        # Timing
        # -------------------------------------------------------------------

        print("=" * 70)
        print("PERFORMANCE")
        print("=" * 70)

        print(
            f"Inference time: {elapsed_seconds:.2f} seconds"
        )

        if elapsed_seconds < 30:
            performance = "GOOD"
        elif elapsed_seconds < 60:
            performance = "ACCEPTABLE"
        elif elapsed_seconds < 120:
            performance = "SLOW"
        else:
            performance = "VERY SLOW"

        print(f"Performance:    {performance}")

        # -------------------------------------------------------------------
        # Runtime metadata
        # -------------------------------------------------------------------

        print()
        print("=" * 70)
        print("RUNTIME INFORMATION")
        print("=" * 70)

        print(f"Runtime: {response.runtime_name}")
        print(f"Model:   {response.model}")

        if response.metadata:
            print()
            print("Metadata:")

            for key, value in response.metadata.items():
                print(f"  {key}: {value}")

        # -------------------------------------------------------------------
        # GPU reminder
        # -------------------------------------------------------------------

        print()
        print("=" * 70)
        print("BENCHMARK INTERPRETATION")
        print("=" * 70)

        print(
            "Your current Ollama process reports approximately "
            "96% CPU / 4% GPU."
        )

        print(
            "The MX350 has 2 GB VRAM, so Gemma 3 4B cannot "
            "fully reside in GPU memory."
        )

        print()

        if elapsed_seconds >= 120:
            print(
                "RESULT: Local Gemma 3 4B vision is too slow "
                "for direct per-frame video processing on "
                "the current hardware."
            )
        elif elapsed_seconds >= 60:
            print(
                "RESULT: Local vision works, but latency is "
                "high for large-scale video processing."
            )
        else:
            print(
                "RESULT: Local vision latency is reasonable "
                "enough for development testing."
            )

        print()
        print("=" * 70)
        print("OLLAMA VISION FRAME BENCHMARK COMPLETE")
        print("=" * 70)

    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())