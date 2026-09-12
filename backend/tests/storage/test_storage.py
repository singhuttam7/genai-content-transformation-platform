import asyncio
from uuid import uuid4

from app.storage.files.local import LocalStorageProvider
from app.storage.schemas import (
    StorageUpload,
    StorageObjectType,
)


async def test():
    storage = LocalStorageProvider()

    object_id = uuid4()

    content = (
        b"GenAI Content Transformation Platform - storage test"
    )

    upload = StorageUpload(
        object_id=object_id,
        object_type=StorageObjectType.SOURCE_FILE,
        filename="storage-test.txt",
        content_type="text/plain",
    )

    result = await storage.upload(upload, content)

    print("Upload: OK")
    print("Key:", result.storage_key)
    print("Size:", result.size_bytes)
    print("Hash:", result.content_hash)

    exists = await storage.exists(result.storage_key)
    print("Exists:", exists)

    downloaded = await storage.download(result.storage_key)

    print(
        "Download:",
        "OK" if downloaded == content else "FAILED",
    )

    uri = await storage.get_uri(result.storage_key)
    print("URI:", uri)

    await storage.delete(result.storage_key)
    print("Delete: OK")

    exists_after_delete = await storage.exists(
        result.storage_key
    )

    print(
        "Exists after delete:",
        exists_after_delete,
    )


if __name__ == "__main__":
    asyncio.run(test())
