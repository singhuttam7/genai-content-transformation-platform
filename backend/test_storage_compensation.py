import asyncio
from uuid import uuid4

from app.storage.compensation import StorageCompensationService
from app.storage.dependencies import get_storage_service


async def test_storage_compensation() -> None:
    storage = get_storage_service()

    compensation = StorageCompensationService(
        storage_service=storage,
    )

    object_id = uuid4()
    filename = "compensation-test.txt"
    content = b"Storage compensation integration test."

    storage_object = await storage.upload_source(
        object_id=object_id,
        filename=filename,
        content_type="text/plain",
        content=content,
        metadata={
            "test": True,
            "purpose": "compensation",
        },
    )

    print("Initial upload: OK")

    assert storage_object.storage_key is not None

    exists_before = await storage.exists(
        storage_object.storage_key
    )

    assert exists_before is True

    print("Storage object exists before compensation: OK")

    await compensation.compensate_upload(
        storage_object.storage_key
    )

    print("Compensation operation: OK")

    exists_after = await storage.exists(
        storage_object.storage_key
    )

    assert exists_after is False

    print("Storage object removed after compensation: OK")

    print()
    print(
        "Storage compensation: ALL TESTS PASSED"
    )


if __name__ == "__main__":
    asyncio.run(
        test_storage_compensation()
    )