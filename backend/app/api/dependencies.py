from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.storage.dependencies import get_storage_service
from app.storage.service import StorageService


DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db_session),
]


StorageServiceDependency = Annotated[
    StorageService,
    Depends(get_storage_service),
]