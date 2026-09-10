import asyncio
from uuid import uuid4

from app.storage.files.local import LocalStorageProvider
from app.storage.schemas import StorageObjectType
from app.storage.service import StorageService


async def test_storage_service():
    provider = LocalStorageProvider()
    storage = StorageService(provider)

    object_id = uuid4()

    content = b"StorageService integration test"

    # ---------------------------------------------------------
    # Upload source
    # ---------------------------------------------------------

    result = await storage.upload_source(
        object_id=object_id,
        filename="service-test.txt",
        content_type="text/plain",
        content=content,
        metadata={
            "test": True,
            "component": "storage_service",
        },
    )

    print("Upload Source: OK")
    print("Object ID:", result.object_id)
    print("Object Type:", result.object_type)
    print("Storage Key:", result.storage_key)
    print("Size:", result.size_bytes)
    print("Hash:", result.content_hash)

    # ---------------------------------------------------------
    # Verify object type
    # ---------------------------------------------------------

    print(
        "Object Type Check:",
        "OK"
        if result.object_type == StorageObjectType.SOURCE_FILE
        else "FAILED",
    )

    # ---------------------------------------------------------
    # Exists
    # ---------------------------------------------------------

    exists = await storage.exists(
        result.storage_key
    )

    print("Exists:", exists)

    # ---------------------------------------------------------
    # Download
    # ---------------------------------------------------------

    downloaded = await storage.download(
        result.storage_key
    )

    print(
        "Download:",
        "OK"
        if downloaded == content
        else "FAILED",
    )

    # ---------------------------------------------------------
    # URI
    # ---------------------------------------------------------

    uri = await storage.get_uri(
        result.storage_key
    )

    print("URI:", uri)

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    await storage.delete(
        result.storage_key
    )

    print("Delete: OK")

    # ---------------------------------------------------------
    # Verify deletion
    # ---------------------------------------------------------

    exists_after_delete = await storage.exists(
        result.storage_key
    )

    print(
        "Exists After Delete:",
        exists_after_delete,
    )


if __name__ == "__main__":
    asyncio.run(test_storage_service())