from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from app.api.dependencies import DatabaseSession, StorageServiceDependency
from app.ingestion.application_service import IngestionApplicationService
from app.ingestion.schemas import IngestionRequest
from app.ingestion.results import IngestionResult
from app.schemas.source import SourceCreateRequest, SourceResponse


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
        ingestion_request = _to_ingestion_request(request)

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