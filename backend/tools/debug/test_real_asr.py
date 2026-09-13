import asyncio
from pathlib import Path

from app.ingestion.speech.faster_whisper import FasterWhisperASRProvider
from app.ingestion.speech.schemas import ASRRequest


async def main():
    provider = FasterWhisperASRProvider(
        model_name="small",
        device="cpu",
        compute_type="int8",
    )

    audio_path = Path(r".\test_data\asr\sample_en.wav")
    audio = audio_path.read_bytes()

    request = ASRRequest(audio=audio)

    print("Starting transcription...")
    result = await provider.transcribe(request)

    print("STATUS:", result.status)
    print("LANGUAGE:", result.language)
    print("PROVIDER:", result.provider)
    print("TEXT:", result.text)
    print("SEGMENTS:", len(result.segments))

    for segment in result.segments:
        print(
            f"[{segment.start_time:.2f}s -> "
            f"{segment.end_time:.2f}s] {segment.text}"
        )


if __name__ == "__main__":
    asyncio.run(main())