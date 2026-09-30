from __future__ import annotations

import json
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api.dependencies import (
    DatabaseSession,
    StorageServiceDependency,
)
from app.ingestion.application_service import (
    IngestionApplicationService,
)
from app.ingestion.schemas import (
    IngestionRequest,
    InputType,
    ProcessingStatus,
)
from app.ingestion.results import (
    IngestionResult,
)
from app.models.source import Source
from app.schemas.source import (
    SourceCreateRequest,
    SourceListItem,
    SourceListResponse,
    SourceResponse,
)


router = APIRouter(
    prefix="/sources",
    tags=["Sources"],
)


def _to_ingestion_request(
    request: SourceCreateRequest,
) -> IngestionRequest:
    return IngestionRequest(
        project_id=request.project_id,
        source_id=request.source_id,
        input_type=request.input_type,
        title=request.title,
        filename=request.filename,
        mime_type=request.mime_type,
        content=request.content,
        url=request.url,
        storage_key=request.storage_key,
        storage_uri=request.storage_uri,
        metadata=request.metadata,
    )


def _to_response(
    result: IngestionResult,
) -> SourceResponse:
    canonical = result.canonical_content

    return SourceResponse(
        source_id=result.source_id,
        status=result.status,
        storage_key=result.storage_key,
        storage_uri=result.storage_uri,
        content_hash=result.content_hash,
        title=canonical.title,
        source_type=canonical.source.source_type,
        canonical_text=canonical.text,
        segments=canonical.segments,
        metadata={
            **canonical.metadata,
            **result.metadata,
        },
        provenance=canonical.provenance,
    )


def _to_list_item(
    source: Source,
) -> SourceListItem:
    try:
        source_type = InputType(
            source.source_type,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unknown source type persisted for "
            f"source {source.id}: {source.source_type}"
        ) from exc

    try:
        source_status = ProcessingStatus(
            source.status,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unknown source status persisted for "
            f"source {source.id}: {source.status}"
        ) from exc

    return SourceListItem(
        id=source.id,
        project_id=source.project_id,
        source_type=source_type,
        title=source.title,
        original_filename=source.original_filename,
        mime_type=source.mime_type,
        storage_uri=source.storage_uri,
        content_hash=source.content_hash,
        metadata=source.source_metadata or {},
        status=source_status,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


@router.get(
    "",
    response_model=SourceListResponse,
)
async def list_sources(
    db: DatabaseSession,
    project_id: UUID = Query(...),
    source_type: InputType | None = Query(None),
    source_status: ProcessingStatus | None = Query(None),
    offset: int = Query(
        0,
        ge=0,
    ),
    limit: int = Query(
        50,
        ge=1,
        le=100,
    ),
) -> SourceListResponse:
    """
    List source records for a project.

    The endpoint intentionally returns lightweight source
    metadata only. Extracted canonical content and segments
    are available through the source creation/detail flow
    and are not included in collection responses.
    """

    filters = [
        Source.project_id == project_id,
    ]

    if source_type is not None:
        filters.append(
            Source.source_type == source_type.value,
        )

    if source_status is not None:
        filters.append(
            Source.status == source_status.value,
        )

    total_result = await db.execute(
        select(func.count(Source.id)).where(
            *filters,
        )
    )

    total = total_result.scalar_one()

    result = await db.execute(
        select(Source)
        .where(*filters)
        .order_by(
            Source.created_at.desc(),
            Source.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    sources = result.scalars().all()

    try:
        items = [
            _to_list_item(source)
            for source in sources
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return SourceListResponse(
        items=items,
        total=total,
    )


@router.post(
    "",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_source(
    request: SourceCreateRequest,
    db: DatabaseSession,
    storage: StorageServiceDependency,
) -> SourceResponse:
    try:
        ingestion_request = _to_ingestion_request(
            request,
        )

    except ValidationError as exc:
        errors = []

        for error in exc.errors():
            errors.append(
                {
                    "type": error.get("type"),
                    "loc": error.get("loc", ()),
                    "msg": error.get("msg"),
                }
            )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=errors,
        ) from exc

    service = IngestionApplicationService(
        session=db,
        storage=storage,
    )

    try:
        result = await service.ingest(
            request=ingestion_request,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return _to_response(result)


@router.post(
    "/upload",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_source(
    db: DatabaseSession,
    storage: StorageServiceDependency,
    file: UploadFile = File(...),
    project_id: UUID = Form(...),
    title: str | None = Form(None),
    metadata: str = Form("{}"),
) -> SourceResponse:
    """
    Upload and ingest a PDF source.

    This endpoint intentionally starts with PDF support.
    Additional document and media types will be added separately.
    """

    filename = file.filename or "source.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Only PDF files are supported by "
                "this upload endpoint."
            ),
        )

    try:
        parsed_metadata = json.loads(metadata)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="metadata must contain valid JSON.",
        ) from exc

    if not isinstance(parsed_metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="metadata must be a JSON object.",
        )

    try:
        content = await file.read()

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded PDF is empty.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is not a valid PDF.",
        )

    upload_metadata = {
        **parsed_metadata,
        "original_filename": filename,
        "uploaded_content_type": file.content_type,
    }

    ingestion_request = IngestionRequest(
        project_id=project_id,
        input_type=InputType.PDF,
        title=title or filename,
        filename=filename,
        mime_type="application/pdf",
        content=content,
        metadata=upload_metadata,
    )

    service = IngestionApplicationService(
        session=db,
        storage=storage,
    )

    try:
        result = await service.ingest(
            request=ingestion_request,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return _to_response(result)