import asyncio

from app.ingestion.parsers.text import TextProcessor
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


async def test_text_processor():
    processor = TextProcessor()

    request = IngestionRequest(
        input_type=InputType.TEXT,
        title="Test Article",
        filename="article.txt",
        mime_type="text/plain",
        content=(
            "Artificial intelligence is transforming "
            "content creation."
        ),
    )

    result = await processor.process(request)

    print("Processor Type:", type(processor).__name__)

    print(
        "Supported TEXT:",
        InputType.TEXT in processor.supported_types,
    )

    print(
        "Supported PROMPT:",
        InputType.PROMPT in processor.supported_types,
    )

    print(
        "Extracted Text:",
        result.text,
    )

    print(
        "Block Count:",
        len(result.blocks),
    )

    print(
        "Block Type:",
        result.blocks[0].block_type.value,
    )

    print(
        "Text Processing:",
        "OK"
        if result.text
        == request.content
        else "FAILED",
    )

    print(
        "Block Processing:",
        "OK"
        if result.blocks[0].block_type
        == ContentBlockType.PARAGRAPH
        else "FAILED",
    )


if __name__ == "__main__":
    asyncio.run(test_text_processor())