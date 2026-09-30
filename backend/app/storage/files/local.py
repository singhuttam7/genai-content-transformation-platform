from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import UUID

from app.core.config import settings
from app.storage.schemas import (
    StorageObject,
    StorageObjectType,
    StorageUpload,
)


class LocalStorageProvider:
    """Filesystem-backed implementation of the storage provider."""

    def __init__(
        self,
        root: str | Path | None = None,
    ) -> None:
        self.root = Path(
            root or settings.storage_root,
        ).resolve()

        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Create the logical storage directories."""

        for object_type in StorageObjectType:
            (
                self.root / object_type.value
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Validate and sanitize a storage filename."""

        if not filename:
            raise ValueError(
                "Filename cannot be empty"
            )

        # Normalize Windows path separators.
        normalized = filename.replace("\\", "/")

        # Split the supplied value into path components.
        path_parts = normalized.split("/")

        # Explicitly reject path traversal.
        if ".." in path_parts:
            raise ValueError(
                "Path traversal is not allowed"
            )

        # Only a filename is allowed.
        # Directory paths must not be supplied here.
        if len(path_parts) != 1:
            raise ValueError(
                "Directory paths are not allowed in filenames"
            )

        name = path_parts[0]

        # Reject current/parent directory references.
        if name in {".", ".."}:
            raise ValueError(
                "Invalid filename"
            )

        # Reject Windows drive-style paths.
        if re.match(r"^[A-Za-z]:", name):
            raise ValueError(
                "Absolute paths are not allowed"
            )

        # Keep only safe filename characters.
        name = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            name,
        )

        if not name:
            raise ValueError(
                "Invalid filename"
            )

        return name

    def _build_path(
        self,
        object_type: StorageObjectType,
        object_id: UUID,
        filename: str,
    ) -> Path:
        """Build a safe filesystem path for a stored object."""

        safe_filename = self._sanitize_filename(
            filename
        )

        object_directory = (
            self.root
            / object_type.value
            / str(object_id)
        )

        path = (
            object_directory
            / safe_filename
        )

        # Defense-in-depth:
        # ensure the resolved path remains inside
        # the configured storage root.
        try:
            path.resolve().relative_to(
                self.root
            )
        except ValueError as exc:
            raise ValueError(
                "Resolved storage path escapes storage root."
            ) from exc

        return path

    @staticmethod
    def _calculate_hash(content: bytes) -> str:
        """Calculate a SHA-256 hash for content."""

        return hashlib.sha256(
            content
        ).hexdigest()

    @staticmethod
    def _build_uri(path: Path) -> str:
        """Return a file URI for a stored object."""

        return path.as_uri()

    async def upload(
        self,
        upload: StorageUpload,
        content: bytes,
    ) -> StorageObject:
        """Store content locally and return storage metadata."""

        # ---------------------------------------------------------
        # 1. Validate file size
        # ---------------------------------------------------------

        max_size = (
            settings.max_upload_size_mb
            * 1024
            * 1024
        )

        if len(content) > max_size:
            raise ValueError(
                "File exceeds maximum allowed size of "
                f"{settings.max_upload_size_mb} MB."
            )

        # ---------------------------------------------------------
        # 2. Calculate content hash
        # ---------------------------------------------------------

        content_hash = self._calculate_hash(
            content
        )

        # ---------------------------------------------------------
        # 3. Verify supplied hash if present
        # ---------------------------------------------------------

        if (
            upload.content_hash is not None
            and upload.content_hash != content_hash
        ):
            raise ValueError(
                "Provided content hash does not "
                "match uploaded content."
            )

        # ---------------------------------------------------------
        # 4. Build safe storage path
        # ---------------------------------------------------------

        path = self._build_path(
            upload.object_type,
            upload.object_id,
            upload.filename,
        )

        # ---------------------------------------------------------
        # 5. Create object directory
        # ---------------------------------------------------------

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # 6. Write through temporary file
        # ---------------------------------------------------------

        temporary_path = path.with_suffix(
            f"{path.suffix}.tmp"
        )

        temporary_path.write_bytes(
            content
        )

        # Atomic replacement on the same filesystem.
        temporary_path.replace(path)

        # ---------------------------------------------------------
        # 7. Build storage metadata
        # ---------------------------------------------------------

        storage_key = str(
            path.relative_to(
                self.root
            ).as_posix()
        )

        return StorageObject(
            object_id=upload.object_id,
            object_type=upload.object_type,
            storage_key=storage_key,
            uri=self._build_uri(path),
            filename=path.name,
            content_type=upload.content_type,
            size_bytes=len(content),
            content_hash=content_hash,
            metadata=upload.metadata,
        )

    async def download(
        self,
        storage_key: str,
    ) -> bytes:
        """Retrieve an object by storage key."""

        path = self._resolve_storage_key(
            storage_key
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Storage object not found: "
                f"{storage_key}"
            )

        return path.read_bytes()

    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        """Check whether a storage object exists."""

        path = self._resolve_storage_key(
            storage_key
        )

        return path.is_file()

    async def delete(
        self,
        storage_key: str,
    ) -> None:
        """Delete a storage object if it exists."""

        path = self._resolve_storage_key(
            storage_key
        )

        if path.is_file():
            path.unlink()

    async def get_uri(
        self,
        storage_key: str,
    ) -> str:
        """Return the URI for an existing storage object."""

        path = self._resolve_storage_key(
            storage_key
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Storage object not found: "
                f"{storage_key}"
            )

        return self._build_uri(path)

    def _resolve_storage_key(
        self,
        storage_key: str,
    ) -> Path:
        """Resolve a storage key safely.

        Prevents storage keys from escaping the
        configured storage root.
        """

        if not storage_key:
            raise ValueError(
                "Storage key cannot be empty"
            )

        # Normalize Windows separators.
        normalized = storage_key.replace(
            "\\",
            "/",
        )

        # Reject obvious traversal components.
        parts = normalized.split("/")

        if ".." in parts:
            raise ValueError(
                "Path traversal is not allowed"
            )

        candidate = (
            self.root / normalized
        ).resolve()

        # Defense-in-depth root containment check.
        try:
            candidate.relative_to(
                self.root
            )
        except ValueError as exc:
            raise ValueError(
                "Storage key escapes storage root."
            ) from exc

        return candidate