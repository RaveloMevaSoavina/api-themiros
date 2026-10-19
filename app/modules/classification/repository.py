"""File de vectorisation des cadres validés, avec la clé serveur.

Les écritures passent par les fonctions SQL de la migration
`20261019000003_segment_classification.sql`, conditionnées au bail.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.repository import vector_literal


@dataclass(frozen=True, slots=True)
class Probe:
    """Texte d'un pilier (`pillar_variable_id` vide) ou d'une variable."""

    pillar_id: UUID
    pillar_variable_id: UUID | None
    text: str
    source_hash: str


@dataclass(frozen=True, slots=True)
class ClaimedFramework:
    framework_id: UUID
    workspace_id: UUID
    lease_token: UUID
    probes: list[Probe]

    @classmethod
    def from_row(cls, row: Any) -> "ClaimedFramework | None":
        if not isinstance(row, dict) or not row.get("lease_token"):
            return None
        return cls(
            framework_id=UUID(row["framework_id"]),
            workspace_id=UUID(row["workspace_id"]),
            lease_token=UUID(row["lease_token"]),
            probes=[
                Probe(
                    pillar_id=UUID(item["pillar_id"]),
                    pillar_variable_id=(
                        UUID(item["pillar_variable_id"])
                        if item.get("pillar_variable_id")
                        else None
                    ),
                    text=item["text"],
                    source_hash=item["source_hash"],
                )
                for item in row.get("probes") or []
            ],
        )


class ClassificationRepository:
    def __init__(self, gateway: SupabaseGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    async def claim(self, worker_id: str, model: str) -> ClaimedFramework | None:
        row = await self._gateway.service_rpc(
            "claim_framework_classification",
            payload={
                "p_worker_id": worker_id,
                "p_model": model,
                "p_lease_seconds": self._settings.ingestion_worker_lease_seconds,
            },
        )
        return ClaimedFramework.from_row(row)

    async def complete(
        self,
        job: ClaimedFramework,
        model: str,
        vectors: list[tuple[Probe, list[float]]],
    ) -> dict[str, Any]:
        # Classer tout le corpus de l'espace peut prendre du temps.
        result = await self._gateway.service_rpc(
            "complete_framework_classification",
            payload={
                "p_framework_id": str(job.framework_id),
                "p_lease_token": str(job.lease_token),
                "p_model": model,
                "p_probes": [
                    {
                        "pillar_id": str(probe.pillar_id),
                        "pillar_variable_id": (
                            str(probe.pillar_variable_id)
                            if probe.pillar_variable_id
                            else None
                        ),
                        "source_hash": probe.source_hash,
                        "embedding": vector_literal(vector),
                    }
                    for probe, vector in vectors
                ],
            },
            request_timeout=120,
        )
        return result if isinstance(result, dict) else {}

    async def fail(self, job: ClaimedFramework, code: str, message: str) -> None:
        await self._gateway.service_rpc(
            "fail_framework_classification",
            payload={
                "p_framework_id": str(job.framework_id),
                "p_lease_token": str(job.lease_token),
                "p_error_code": code,
                "p_error_message": message,
            },
        )
