"""Accès aux données du pipeline, avec la clé serveur.

Toutes les écritures passent par les fonctions SQL de la migration
`20261006000001_document_ingestion.sql`, conditionnées au bail du job.
"""

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.analysis.segmenter import PageText
from app.modules.ingestion.domain import IngestionRules, WorkspaceProfile


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    id: UUID
    document_id: UUID
    workspace_id: UUID
    kind: str
    lease_token: UUID
    attempts: int
    max_attempts: int

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "ClaimedJob | None":
        if not isinstance(row, dict) or not row.get("id") or not row.get("lease_token"):
            return None
        return cls(
            id=UUID(row["id"]),
            document_id=UUID(row["document_id"]),
            workspace_id=UUID(row["workspace_id"]),
            kind=row["kind"],
            lease_token=UUID(row["lease_token"]),
            attempts=int(row.get("attempts") or 1),
            max_attempts=int(row.get("max_attempts") or 1),
        )


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: UUID
    workspace_id: UUID
    original_filename: str
    storage_path: str
    file_hash: str
    mime_type: str
    status: str
    status_reason_code: str | None
    detected_language: str | None


@dataclass(frozen=True, slots=True)
class StoredPages:
    pages: list[PageText]
    total_pages: int
    exploitable_pages: int


class IngestionRepository:
    def __init__(self, gateway: SupabaseGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    # --- File de traitement -------------------------------------------------

    async def claim(self, worker_id: str) -> ClaimedJob | None:
        row = await self._gateway.service_rpc(
            "claim_ingestion_job",
            payload={
                "p_worker_id": worker_id,
                "p_lease_seconds": self._settings.ingestion_worker_lease_seconds,
            },
        )
        return ClaimedJob.from_row(row)

    async def start(
        self, document_id: UUID, kind: str, worker_id: str, requested_by: UUID
    ) -> ClaimedJob:
        row = await self._gateway.service_rpc(
            "start_ingestion_job",
            payload={
                "p_document_id": str(document_id),
                "p_kind": kind,
                "p_worker_id": worker_id,
                "p_requested_by": str(requested_by),
                "p_lease_seconds": self._settings.ingestion_worker_lease_seconds,
            },
        )
        job = ClaimedJob.from_row(row)
        if job is None:
            raise ApiError(
                status_code=503,
                code="DATABASE_UNAVAILABLE",
                detail="La tâche d'ingestion n'a pas pu être démarrée",
            )
        return job

    async def enqueue(
        self, document_id: UUID, kind: str, requested_by: UUID | None
    ) -> dict[str, Any]:
        row = await self._gateway.service_rpc(
            "enqueue_ingestion_job",
            payload={
                "p_document_id": str(document_id),
                "p_kind": kind,
                "p_requested_by": str(requested_by) if requested_by else None,
            },
        )
        if not isinstance(row, dict) or not row.get("id"):
            raise ApiError(
                status_code=503,
                code="QUEUE_UNAVAILABLE",
                detail="La tâche d'ingestion n'a pas pu être créée",
            )
        return row

    async def heartbeat(self, job: ClaimedJob, stage: str | None = None) -> bool:
        result = await self._gateway.service_rpc(
            "heartbeat_ingestion_job",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_stage": stage,
                "p_lease_seconds": self._settings.ingestion_worker_lease_seconds,
            },
        )
        return result is True

    async def fail(
        self, job: ClaimedJob, code: str, message: str, *, retry: bool
    ) -> str:
        return await self._gateway.service_rpc(
            "fail_ingestion_job",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_error_code": code,
                "p_error_message": message,
                "p_retry": retry,
            },
        )

    async def complete(self, job: ClaimedJob, result: dict[str, Any]) -> None:
        await self._gateway.service_rpc(
            "complete_ingestion_job",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_result": result,
            },
        )

    # --- Lecture du contexte ------------------------------------------------

    async def load_document(self, document_id: UUID) -> DocumentRecord:
        rows = await self._gateway.service_select(
            "documents",
            params={
                "id": f"eq.{document_id}",
                "deleted_at": "is.null",
                "select": (
                    "id,workspace_id,original_filename,storage_path,file_hash,"
                    "mime_type,status,status_reason_code,detected_language"
                ),
                "limit": "1",
            },
        )
        if not rows:
            raise ApiError(
                status_code=404,
                code="DOCUMENT_NOT_FOUND",
                detail="Document introuvable",
            )
        row = rows[0]
        return DocumentRecord(
            id=UUID(row["id"]),
            workspace_id=UUID(row["workspace_id"]),
            original_filename=row["original_filename"],
            storage_path=row["storage_path"],
            file_hash=row["file_hash"],
            mime_type=row["mime_type"],
            status=row["status"],
            status_reason_code=row.get("status_reason_code"),
            detected_language=row.get("detected_language"),
        )

    async def load_profile(
        self, workspace_id: UUID
    ) -> tuple[WorkspaceProfile, list[str]]:
        """Empreinte de l'espace, et termes à rechercher pour ses financeurs
        (noms, sigles et alias du référentiel)."""
        rows = await self._gateway.service_select(
            "workspaces",
            params={
                "id": f"eq.{workspace_id}",
                "select": (
                    "id,kind,target_country,financiers,themes,expected_languages,"
                    "declared_stage,start_year,end_year"
                ),
                "limit": "1",
            },
        )
        if not rows:
            raise ApiError(
                status_code=404,
                code="WORKSPACE_NOT_FOUND",
                detail="Workspace introuvable",
            )
        row = rows[0]
        financiers = [
            item for item in row.get("financiers") or [] if isinstance(item, dict)
        ]
        names = [str(item.get("name") or item.get("code") or "") for item in financiers]
        codes = sorted(
            {
                str(item["code"])
                for item in financiers
                if item.get("code") and item.get("code") != "OTHER"
            }
        )
        terms = {name for name in names if name} | set(codes)
        if codes:
            references = await self._gateway.service_select(
                "ref_frameworks",
                params={
                    "financier_code": f"in.({','.join(codes)})",
                    "select": "financier_code,financier_aliases",
                },
            )
            for reference in references:
                terms.update(reference.get("financier_aliases") or [])
        profile = WorkspaceProfile(
            id=str(row["id"]),
            kind=str(row.get("kind") or "programme"),
            target_country=str(row.get("target_country") or "").upper(),
            financiers=tuple(name for name in names if name),
            themes=tuple(row.get("themes") or ()),
            expected_languages=tuple(row.get("expected_languages") or ("fr", "en")),
            stage=str(row.get("declared_stage") or ""),
            start_year=row.get("start_year"),
            end_year=row.get("end_year"),
        )
        return profile, sorted(term for term in terms if term)

    async def load_rules(self, workspace_id: UUID) -> IngestionRules:
        """Paramètres propres à l'espace, à défaut les valeurs par défaut."""
        rows = await self._gateway.service_select(
            "ingestion_settings",
            params={
                "select": "*",
                "or": f"(workspace_id.eq.{workspace_id},workspace_id.is.null)",
                "order": "workspace_id.nullslast,version.desc",
                "limit": "1",
            },
        )
        return IngestionRules.from_row(rows[0]) if rows else IngestionRules()

    async def load_overrides(self, document_id: UUID) -> dict[str, str]:
        """Langue ou pays corrigés par un humain (UC15)."""
        rows = await self._gateway.service_select(
            "document_metadata",
            params={
                "document_id": f"eq.{document_id}",
                "source": "eq.user",
                "key": "in.(language,country)",
                "select": "key,value",
            },
        )
        return {row["key"]: row["value"] for row in rows if row.get("value")}

    async def load_pages(self, document_id: UUID) -> StoredPages:
        rows = await self._gateway.service_select(
            "document_pages",
            params={
                "document_id": f"eq.{document_id}",
                "select": "page_number,text,section_title,char_start,is_exploitable",
                "order": "page_number.asc",
            },
        )
        if not rows:
            raise ApiError(
                status_code=409,
                code="DOCUMENT_NOT_EXTRACTED",
                detail="Le texte du document n'a pas encore été extrait",
            )
        return StoredPages(
            pages=[
                PageText(
                    page_number=row["page_number"],
                    text=row.get("text") or "",
                    char_start=row.get("char_start") or 0,
                    section_title=row.get("section_title"),
                )
                for row in rows
            ],
            total_pages=len(rows),
            exploitable_pages=sum(bool(row.get("is_exploitable")) for row in rows),
        )

    async def download(self, storage_path: str) -> bytes:
        return await self._gateway.storage_download(
            self._settings.ingestion_storage_bucket, storage_path
        )

    async def load_fingerprint(
        self, workspace_id: UUID
    ) -> tuple[str, list[float]] | None:
        rows = await self._gateway.service_select(
            "semantic_fingerprints",
            params={
                "workspace_id": f"eq.{workspace_id}",
                "select": "source_hash,embedding",
                "limit": "1",
            },
        )
        if not rows or not rows[0].get("source_hash") or not rows[0].get("embedding"):
            return None
        embedding = rows[0]["embedding"]
        vector = json.loads(embedding) if isinstance(embedding, str) else embedding
        return rows[0]["source_hash"], [float(value) for value in vector]

    async def save_fingerprint(
        self,
        profile: WorkspaceProfile,
        *,
        source_text: str,
        source_hash: str,
        vector: list[float],
        model: str,
    ) -> None:
        await self._gateway.service_upsert(
            "semantic_fingerprints",
            on_conflict="workspace_id",
            payload={
                "workspace_id": profile.id,
                "country": profile.target_country,
                "financiers": list(profile.financiers),
                "themes": list(profile.themes),
                "expected_languages": list(profile.expected_languages),
                "stage": profile.stage or "implementation",
                "start_year": profile.start_year or 0,
                "end_year": profile.end_year or 0,
                "embedding": vector_literal(vector),
                "source_text": source_text,
                "source_hash": source_hash,
                "model": model,
            },
        )

    # --- Écritures du pipeline ----------------------------------------------

    async def commit_extraction(
        self,
        job: ClaimedJob,
        *,
        pages: list[dict[str, Any]],
        pdf_type: str | None,
        metadata: dict[str, Any],
    ) -> None:
        await self._gateway.service_rpc(
            "commit_document_extraction",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_pages": pages,
                "p_pdf_type": pdf_type,
                "p_metadata": metadata,
            },
            request_timeout=60,
        )

    async def commit_qualification(
        self, job: ClaimedJob, result: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._gateway.service_rpc(
            "commit_document_qualification",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_result": result,
            },
        )

    async def append_segments(
        self, job: ClaimedJob, generation: UUID, rows: list[dict[str, Any]]
    ) -> int:
        return await self._gateway.service_rpc(
            "append_document_segments",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_generation": str(generation),
                "p_segments": rows,
            },
            request_timeout=60,
        )

    async def publish_index(self, job: ClaimedJob, generation: UUID, model: str) -> int:
        return await self._gateway.service_rpc(
            "publish_document_index",
            payload={
                "p_job_id": str(job.id),
                "p_lease_token": str(job.lease_token),
                "p_generation": str(generation),
                "p_model": model,
            },
        )

    async def unpublish_index(self, job: ClaimedJob) -> None:
        await self._gateway.service_rpc(
            "unpublish_document_index",
            payload={"p_job_id": str(job.id), "p_lease_token": str(job.lease_token)},
        )


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
