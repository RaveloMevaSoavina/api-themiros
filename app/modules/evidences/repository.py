from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.supabase import SupabaseGateway


class EvidenceRepository:
    def __init__(self, gateway: SupabaseGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    async def evidence(self, evidence_id: UUID, *, token: str) -> dict[str, Any] | None:
        """Lue avec le jeton de l'utilisateur : la RLS des preuves et des
        documents décide seule de ce qu'il peut ouvrir."""
        rows = await self._gateway.rest_get(
            "evidences",
            token=token,
            params={
                "select": (
                    "id,page_number,excerpt,"
                    "documents(original_filename,storage_path,mime_type)"
                ),
                "id": f"eq.{evidence_id}",
            },
        )
        return rows[0] if rows else None

    async def download(self, storage_path: str) -> bytes:
        return await self._gateway.storage_download(
            self._settings.ingestion_storage_bucket, storage_path
        )
