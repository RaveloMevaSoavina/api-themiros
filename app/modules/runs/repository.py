"""File des runs d'analyse, avec la clé serveur.

Les écritures passent par les fonctions SQL des migrations
`20261020000002_analysis_runs.sql` et `20261020000003_run_results.sql`,
conditionnées au bail.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.runs.domain import RunInputs


@dataclass(frozen=True, slots=True)
class ClaimedRun:
    run_id: UUID
    workspace_id: UUID
    lease_token: UUID
    attempts: int
    rules_applied: dict[str, Any]

    @classmethod
    def from_row(cls, row: Any) -> "ClaimedRun | None":
        if not isinstance(row, dict) or not row.get("lease_token"):
            return None
        return cls(
            run_id=UUID(row["run_id"]),
            workspace_id=UUID(row["workspace_id"]),
            lease_token=UUID(row["lease_token"]),
            attempts=int(row.get("attempts") or 0),
            rules_applied=row.get("rules_applied") or {},
        )


class RunRepository:
    def __init__(self, gateway: SupabaseGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    async def claim(self, worker_id: str, engine_version: str) -> ClaimedRun | None:
        row = await self._gateway.service_rpc(
            "claim_run",
            payload={
                "p_worker_id": worker_id,
                "p_engine_version": engine_version,
                "p_lease_seconds": self._settings.run_lease_seconds,
            },
        )
        return ClaimedRun.from_row(row)

    async def inputs(self, run: ClaimedRun) -> RunInputs:
        row = await self._gateway.service_rpc(
            "run_inputs", payload=self._lease(run), request_timeout=60
        )
        return RunInputs.from_row(row)

    async def step(
        self,
        run: ClaimedRun,
        step_name: str,
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        await self._gateway.service_rpc(
            "update_run_step",
            payload={
                **self._lease(run),
                "p_step_name": step_name,
                "p_status": status,
                "p_output_summary": summary,
                "p_lease_seconds": self._settings.run_lease_seconds,
            },
        )

    async def complete(
        self, run: ClaimedRun, results: dict[str, Any]
    ) -> dict[str, Any]:
        result = await self._gateway.service_rpc(
            "complete_run",
            payload={**self._lease(run), "p_results": results},
            request_timeout=120,
        )
        return result if isinstance(result, dict) else {}

    async def defer(self, run: ClaimedRun, delay_seconds: int, reason: str) -> None:
        await self._gateway.service_rpc(
            "defer_run",
            payload={
                **self._lease(run),
                "p_delay_seconds": delay_seconds,
                "p_reason": reason,
            },
        )

    async def fail(
        self, run: ClaimedRun, step_name: str | None, code: str, message: str
    ) -> None:
        await self._gateway.service_rpc(
            "fail_run",
            payload={
                **self._lease(run),
                "p_step_name": step_name,
                "p_error_code": code,
                "p_error_message": message,
            },
        )

    @staticmethod
    def _lease(run: ClaimedRun) -> dict[str, str]:
        return {"p_run_id": str(run.run_id), "p_lease_token": str(run.lease_token)}
