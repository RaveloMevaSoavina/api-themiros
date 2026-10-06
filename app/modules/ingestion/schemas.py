from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel


class DocumentRequest(BaseModel):
    workspace_id: UUID
    document_id: UUID


class SegmentRequest(BaseModel):
    document_id: UUID


class FullIngestionRequest(BaseModel):
    workspace_id: UUID
    document_id: UUID
    storage_path: str
    mime_type: str


class ExtractedPageResponse(BaseModel):
    page: int
    text: str
    is_ocr: bool


class ExtractResponse(BaseModel):
    document_id: UUID
    total_pages: int
    exploitable_pages: int
    language: str | None
    language_confidence: float
    pages: list[ExtractedPageResponse]


class QualifyResponse(BaseModel):
    document_id: UUID
    relevance_score: int
    status: str
    detected_language: str | None
    detected_country: str | None
    message: str | None


class SegmentResponse(BaseModel):
    document_id: UUID
    segments_count: int
    embeddings_count: int
    indexed: bool


class FullIngestionResponse(BaseModel):
    job_id: UUID
    status: Literal["queued", "running"]
    poll_url: str


class JobResponse(BaseModel):
    job_id: UUID
    document_id: UUID
    kind: str
    status: str
    stage: str | None
    result: dict[str, Any] | None
    error_code: str | None
    error: str | None
    attempts: int
    max_attempts: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
