import asyncio

from app.ingestion.schemas import (
    IngestionRequest,
    InputType,
)
from app.ingestion.storage_resolver import (
    StorageBackedContentResolver,
)
from app.storage.dependencies import get_storage_service


async def test_inline_bytes() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)

    expected = b"hello binary world"

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="test.pdf",
        mime_type="application/pdf",
        content=expected,
    )

    result = await resolver.resolve(request)

    assert result == expected

    print("Inline binary resolution: OK")


async def test_storage_reference() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)

    expected = b"stored binary content"

    object_id = __import__("uuid").uuid4()

    stored = await storage.upload_source(
        object_id=object_id,
        filename="resolver-test.bin",
        content_type="application/octet-stream",
        content=expected,
    )

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="resolver-test.bin",
        mime_type="application/octet-stream",
        storage_key=stored.storage_key,
    )

    result = await resolver.resolve(request)

    assert result == expected

    await storage.delete(stored.storage_key)

    print("Storage-reference resolution: OK")
    print("Storage cleanup: OK")


async def test_missing_content() -> None:
    storage = get_storage_service()
    resolver = StorageBackedContentResolver(storage)

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="missing.pdf",
        mime_type="application/pdf",
        storage_key=None,
        content=b"",
    )

    # Empty bytes are still a valid representation at this boundary.
    result = await resolver.resolve(request)

    assert result == b""

    print("Empty inline binary resolution: OK")


async def main() -> None:
    print()
    print("==============================================")
    print("CONTENT RESOLVER TESTS")
    print("==============================================")
    print()

    await test_inline_bytes()
    await test_storage_reference()
    await test_missing_content()

    print()
    print("==============================================")
    print("CONTENT RESOLVER TESTS: ALL PASSED")
    print("==============================================")


if __name__ == "__main__":
    asyncio.run(main())