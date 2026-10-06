from typing import Annotated
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.analysis.country import get_country_detector
from app.modules.ingestion.analysis.embeddings import build_embedder
from app.modules.ingestion.pipeline import IngestionPipeline
from app.modules.ingestion.repository import IngestionRepository
from app.modules.ingestion.schemas import (
    DocumentRequest,
    ExtractResponse,
    FullIngestionRequest,
    FullIngestionResponse,
    JobResponse,
    QualifyResponse,
    SegmentRequest,
    SegmentResponse,
)
from app.modules.ingestion.service import IngestionService

router = APIRouter(prefix="/ingestion", tags=["ingestion"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_ingestion_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> IngestionService:
    client: httpx.AsyncClient = request.app.state.http_client
    gateway = SupabaseGateway(settings, client)
    repository = IngestionRepository(gateway, settings)
    pipeline = IngestionPipeline(
        repository,
        build_embedder(settings),
        get_country_detector(settings.ingestion_ner_model),
        settings,
    )
    return IngestionService(gateway, repository, pipeline, settings)


Service = Annotated[IngestionService, Depends(get_ingestion_service)]
User = Annotated[CurrentUser, Depends(get_current_user)]


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(
    file: Annotated[UploadFile, File()],
    workspace_id: Annotated[UUID, Form()],
    document_id: Annotated[UUID, Form()],
    user: User,
    service: Service,
) -> ExtractResponse:
    """Extraction + OCR (catalogue §6.1)."""
    return await service.extract(
        workspace_id=workspace_id,
        document_id=document_id,
        content=await file.read(),
        user=user,
    )


@router.post("/qualify", response_model=QualifyResponse)
async def qualify_document(
    payload: DocumentRequest, user: User, service: Service
) -> QualifyResponse:
    """Score de pertinence et statut (catalogue §6.2)."""
    return await service.qualify(
        workspace_id=payload.workspace_id, document_id=payload.document_id, user=user
    )


@router.post("/segment", response_model=SegmentResponse)
async def segment_document(
    payload: SegmentRequest, user: User, service: Service
) -> SegmentResponse:
    """Segmentation et indexation (catalogue §6.3)."""
    return await service.segment(document_id=payload.document_id, user=user)


@router.post(
    "/full",
    response_model=FullIngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def full_ingestion(
    payload: FullIngestionRequest, user: User, service: Service
) -> FullIngestionResponse:
    """Ingestion complète, asynchrone (catalogue §6.4)."""
    return await service.full(
        workspace_id=payload.workspace_id,
        document_id=payload.document_id,
        storage_path=payload.storage_path,
        mime_type=payload.mime_type,
        user=user,
    )


@jobs_router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, user: User, service: Service) -> JobResponse:
    try:
        parsed = UUID(job_id)
    except ValueError as error:
        raise ApiError(
            status_code=404,
            code="JOB_NOT_FOUND",
            detail="Tâche d'ingestion introuvable",
        ) from error
    return await service.get_job(parsed, user)
