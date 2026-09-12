import asyncio
from uuid import uuid4

from app.ingestion.parsers.document import TextDocumentProcessor
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)
from app.storage.dependencies import get_storage_service
from app.ingestion.storage_resolver import StorageBackedContentResolver


async def test_txt_inline() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)
    processor = TextDocumentProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.TXT,
        filename="report.txt",
        mime_type="text/plain",
        content=(
            b"Artificial intelligence is transforming communication.\n\n"
            b"This is the second paragraph."
        ),
        metadata={"test": True},
    )

    result = await processor.process(request)

    assert result.text == (
        "Artificial intelligence is transforming communication.\n\n"
        "This is the second paragraph."
    )

    assert len(result.blocks) == 2
    assert result.blocks[0].block_type == ContentBlockType.PARAGRAPH
    assert result.blocks[1].block_type == ContentBlockType.PARAGRAPH

    assert result.blocks[0].content == (
        "Artificial intelligence is transforming communication."
    )

    assert result.blocks[1].content == (
        "This is the second paragraph."
    )

    assert result.metadata["test"] is True

    print("TXT inline processing: OK")


async def test_txt_storage_reference() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)
    processor = TextDocumentProcessor(resolver)

    content = (
        b"Stored document content.\n\n"
        b"Second stored paragraph."
    )

    object_id = uuid4()

    stored = await storage.upload_source(
        object_id=object_id,
        filename="stored.txt",
        content_type="text/plain",
        content=content,
    )

    request = IngestionRequest(
        input_type=InputType.TXT,
        filename="stored.txt",
        mime_type="text/plain",
        storage_key=stored.storage_key,
    )

    result = await processor.process(request)

    assert result.text == content.decode("utf-8")
    assert len(result.blocks) == 2

    await storage.delete(stored.storage_key)

    print("TXT storage-reference processing: OK")
    print("TXT storage cleanup: OK")


async def test_markdown_inline() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)
    processor = TextDocumentProcessor(resolver)

    content = (
        b"# Cybersecurity Advisory\n\n"
        b"This advisory provides important security guidance.\n\n"
        b"## Recommended Actions\n\n"
        b"Update affected systems immediately."
    )

    request = IngestionRequest(
        input_type=InputType.MARKDOWN,
        filename="advisory.md",
        mime_type="text/markdown",
        content=content,
    )

    result = await processor.process(request)

    assert result.text == content.decode("utf-8")

    assert len(result.blocks) == 4

    assert result.blocks[0].block_type == ContentBlockType.HEADING
    assert result.blocks[0].content == "Cybersecurity Advisory"

    assert result.blocks[1].block_type == ContentBlockType.PARAGRAPH

    assert result.blocks[2].block_type == ContentBlockType.HEADING
    assert result.blocks[2].content == "Recommended Actions"

    assert result.blocks[3].block_type == ContentBlockType.PARAGRAPH

    assert result.blocks[0].metadata["markdown"] is True

    print("Markdown inline processing: OK")


async def test_invalid_utf8() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)
    processor = TextDocumentProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.TXT,
        filename="invalid.txt",
        mime_type="text/plain",
        content=b"\xff\xfe\xfa",
    )

    try:
        await processor.process(request)
        raise AssertionError(
            "Invalid UTF-8 content was accepted."
        )
    except ValueError as exc:
        assert "UTF-8" in str(exc)

    print("Invalid UTF-8 rejection: OK")


async def test_empty_document() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)
    processor = TextDocumentProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.TXT,
        filename="empty.txt",
        mime_type="text/plain",
        content=b"   \n\n   ",
    )

    try:
        await processor.process(request)
        raise AssertionError(
            "Empty document was accepted."
        )
    except ValueError as exc:
        assert "empty" in str(exc).lower()

    print("Empty document rejection: OK")


async def main() -> None:
    print()
    print("==============================================")
    print("TEXT DOCUMENT PROCESSOR TESTS")
    print("==============================================")
    print()

    await test_txt_inline()
    await test_txt_storage_reference()
    await test_markdown_inline()
    await test_invalid_utf8()
    await test_empty_document()

    print()
    print("==============================================")
    print("TEXT DOCUMENT PROCESSOR: ALL TESTS PASSED")
    print("==============================================")


if __name__ == "__main__":
    asyncio.run(main())