from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.ingestion.video.vision import VisionService
from app.ingestion.video.vision_dependencies import (
    get_vision_service,
)
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

VisionServiceDependency = Annotated[
    VisionService,
    Depends(get_vision_service),
]