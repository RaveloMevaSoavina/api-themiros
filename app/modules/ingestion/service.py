"""Endpoints du catalogue §6 : chaque étape synchrone passe par une tâche
démarrée directement, avec le même bail et les mêmes écritures que le worker.
"""

import contextlib
import socket
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from uuid import UUID

from app.api.dependencies import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.errors import is_abandon
from app.modules.ingestion.pipeline import IngestionPipeline, StageCallback, lease_lost
from app.modules.ingestion.repository import ClaimedJob, IngestionRepository
from app.modules.ingestion.schemas import (
    ExtractedPageResponse,
    ExtractResponse,
    FullIngestionResponse,
    JobResponse,
    QualifyResponse,
    SegmentResponse,
)
from app.modules.ingestion.worker import describe_error

T = TypeVar("T")


class IngestionService:
    def __init__(
        self,
        gateway: SupabaseGateway,
        repository: IngestionRepository,
        pipeline: IngestionPipeline,
        settings: Settings,
    ) -> None:
        self._gateway = gateway
        self._repository = repository
        self._pipeline = pipeline
        self._settings = settings
        self._worker_id = f"api-{socket.gethostname()}"

    async def ensure_document_access(
        self, document_id: UUID, user: CurrentUser, workspace_id: UUID | None = None
    ) -> dict[str, Any]:
        """Lecture avec le jeton de l'utilisateur : la RLS vérifie qu'il est
        membre de l'espace."""
        params = {
            "id": f"eq.{document_id}",
            "deleted_at": "is.null",
            "select": "id,workspace_id,storage_path,mime_type,file_hash",
            "limit": "1",
        }
        if workspace_id:
            params["workspace_id"] = f"eq.{workspace_id}"
        rows = await self._gateway.rest_get(
            "documents", token=user.token, params=params
        )
        if not rows:
            raise ApiError(
                status_code=404,
                code="DOCUMENT_NOT_FOUND",
                detail="Document introuvable dans ce workspace",
            )
        return rows[0]

    async def full(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        storage_path: str,
        mime_type: str,
        user: CurrentUser,
    ) -> FullIngestionResponse:
        document = await self.ensure_document_access(document_id, user, workspace_id)
        if (
            document["storage_path"] != storage_path
            or document["mime_type"] != mime_type
        ):
            raise ApiError(
                status_code=409,
                code="INGESTION_PAYLOAD_MISMATCH",
                detail="Le chemin ou le type du fichier ne correspond pas au document",
            )
        job = await self._repository.enqueue(document_id, "full", user.id)
        return FullIngestionResponse(
            job_id=job["id"],
            status="running" if job.get("status") == "running" else "queued",
            poll_url=f"/api/v1/jobs/{job['id']}",
        )

    async def extract(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        content: bytes,
        user: CurrentUser,
    ) -> ExtractResponse:
        await self.ensure_document_access(document_id, user, workspace_id)
        document = await self._repository.load_document(document_id)
        profile, _ = await self._repository.load_profile(document.workspace_id)
        rules = await self._repository.load_rules(document.workspace_id)

        async def step(job: ClaimedJob, on_stage: StageCallback):
            outcome = await self._pipeline.extract(
                job, document, profile, rules, on_stage, content=content
            )
            response = ExtractResponse(
                document_id=document_id,
                total_pages=outcome.stored.total_pages,
                exploitable_pages=outcome.stored.exploitable_pages,
                language=outcome.language.value,
                language_confidence=outcome.language.confidence,
                pages=[
                    ExtractedPageResponse(
                        page=page.page_number, text=page.text, is_ocr=page.is_ocr
                    )
                    for page in outcome.document.pages
                ],
            )
            return response, {
                "kind": "extract",
                "extraction": response.model_dump(mode="json", exclude={"pages"}),
            }

        return await self._run(document_id, "extract", user, step)

    async def qualify(
        self, *, workspace_id: UUID, document_id: UUID, user: CurrentUser
    ) -> QualifyResponse:
        await self.ensure_document_access(document_id, user, workspace_id)

        async def step(job: ClaimedJob, on_stage: StageCallback):
            result = await self._pipeline.run(job, on_stage)
            qualification = result["qualification"]
            return (
                QualifyResponse(
                    document_id=document_id,
                    relevance_score=qualification["relevance_score"],
                    status=qualification["status"],
                    detected_language=qualification["language"],
                    detected_country=qualification["country"],
                    message=qualification["message"],
                ),
                result,
            )

        return await self._run(document_id, "qualify", user, step)

    async def segment(self, *, document_id: UUID, user: CurrentUser) -> SegmentResponse:
        await self.ensure_document_access(document_id, user)

        async def step(job: ClaimedJob, on_stage: StageCallback):
            result = await self._pipeline.run(job, on_stage)
            count = result["indexation"]["segments_count"]
            return (
                SegmentResponse(
                    document_id=document_id,
                    segments_count=count,
                    embeddings_count=count,
                    indexed=True,
                ),
                result,
            )

        return await self._run(document_id, "index", user, step)

    async def get_job(self, job_id: UUID, user: CurrentUser) -> JobResponse:
        rows = await self._gateway.rest_get(
            "ingestion_jobs",
            token=user.token,
            params={
                "id": f"eq.{job_id}",
                "select": (
                    "id,document_id,kind,status,stage,result,error_code,"
                    "error_message,attempts,max_attempts,created_at,started_at,"
                    "completed_at"
                ),
                "limit": "1",
            },
        )
        if not rows:
            raise ApiError(
                status_code=404,
                code="JOB_NOT_FOUND",
                detail="Tâche d'ingestion introuvable",
            )
        row = rows[0]
        return JobResponse(
            job_id=row["id"],
            document_id=row["document_id"],
            kind=row["kind"],
            status=row["status"],
            stage=row.get("stage"),
            result=row.get("result"),
            error_code=row.get("error_code"),
            error=row.get("error_message"),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
        )

    async def _run(
        self,
        document_id: UUID,
        kind: str,
        user: CurrentUser,
        step: Callable[
            [ClaimedJob, StageCallback], Awaitable[tuple[T, dict[str, Any]]]
        ],
    ) -> T:
        job = await self._repository.start(document_id, kind, self._worker_id, user.id)

        async def on_stage(stage: str) -> None:
            if not await self._repository.heartbeat(job, stage):
                raise lease_lost()

        try:
            value, result = await step(job, on_stage)
            await self._repository.complete(job, result)
            return value
        except Exception as error:
            if not is_abandon(error):
                code, message = describe_error(error)
                with contextlib.suppress(ApiError):
                    await self._repository.fail(job, code, message, retry=False)
            if isinstance(error, ApiError):
                raise
            raise ApiError(
                status_code=500,
                code="INGESTION_FAILED",
                detail="Le traitement du document a échoué",
            ) from error
