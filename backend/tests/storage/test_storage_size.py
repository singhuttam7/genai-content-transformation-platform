import asyncio
from uuid import uuid4

from app.core.config import settings
from app.storage.files.local import LocalStorageProvider
from app.storage.schemas import (
    StorageObjectType,
    StorageUpload,
)


async def test_upload_size_limit():
    # Save original configuration.
    original_limit = settings.max_upload_size_mb

    try:
        # Use a tiny limit for testing.
        settings.max_upload_size_mb = 1

        storage = LocalStorageProvider()

        # Create content larger than 1 MB.
        content = b"x" * (2 * 1024 * 1024)

        upload = StorageUpload(
            object_id=uuid4(),
            object_type=StorageObjectType.SOURCE_FILE,
            filename="large-test.bin",
            content_type="application/octet-stream",
        )

        try:
            await storage.upload(
                upload,
                content,
            )

            print("Upload Size Limit: FAILED")

        except ValueError as exc:
            print("Upload Size Limit: BLOCKED")
            print("Exception:", type(exc).__name__)
            print("Message:", str(exc))

    finally:
        # Restore original configuration.
        settings.max_upload_size_mb = original_limit


if __name__ == "__main__":
    asyncio.run(test_upload_size_limit())