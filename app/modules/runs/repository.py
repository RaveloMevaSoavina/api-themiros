"""File des runs d'analyse, avec la clé serveur.

Les écritures passent par les fonctions SQL des migrations
`20261020000002_analysis_runs.sql`, `20261020000003_run_results.sql`,
`20261021000002_layer_b.sql` et `20261021000005_layer_c.sql`, conditionnées
au bail.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.repository import vector_literal
from app.modules.runs.domain import RunInputs
from app.modules.runs.layer_b import Candidate, LayerBInputs
from app.modules.runs.layer_c import LayerCInputs


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

    async def layer_b_inputs(self, run: ClaimedRun) -> LayerBInputs:
        row = await self._gateway.service_rpc(
            "run_layer_b_inputs", payload=self._lease(run), request_timeout=60
        )
        return LayerBInputs.from_row(row or {})

    async def sub_question_matches(
        self,
        run: ClaimedRun,
        model: str,
        queries: list[tuple[UUID, list[float]]],
        top_k: int,
    ) -> list[Candidate]:
        """Recherche exacte : son coût croît avec le nombre de sous-questions,
        d'où des lots qui tiennent chacun sous le statement_timeout."""
        size = self._settings.run_layer_b_match_batch_size
        candidates: list[Candidate] = []
        for start in range(0, len(queries), size):
            rows = await self._gateway.service_rpc(
                "run_sub_question_matches",
                payload={
                    **self._lease(run),
                    "p_model": model,
                    "p_queries": [
                        {
                            "sub_question_id": str(question_id),
                            "embedding": vector_literal(vector),
                        }
                        for question_id, vector in queries[start : start + size]
                    ],
                    "p_top_k": top_k,
                },
                request_timeout=120,
            )
            candidates.extend(Candidate.from_row(row) for row in rows or [])
        return candidates

    async def layer_c_inputs(self, run: ClaimedRun) -> LayerCInputs:
        row = await self._gateway.service_rpc(
            "run_layer_c_inputs", payload=self._lease(run), request_timeout=60
        )
        return LayerCInputs.from_row(row or {})

    async def segment_texts(
        self, run: ClaimedRun, segment_ids: list[UUID]
    ) -> dict[UUID, str]:
        if not segment_ids:
            return {}
        row = await self._gateway.service_rpc(
            "run_segment_texts",
            payload={
                **self._lease(run),
                "p_segment_ids": [str(segment) for segment in segment_ids],
            },
            request_timeout=60,
        )
        return {UUID(key): text for key, text in (row or {}).items()}

    async def prompt(self, code: str, version: str) -> dict[str, Any]:
        row = await self._gateway.service_rpc(
            "get_prompt_definition",
            payload={"prompt_code": code, "prompt_version": version},
        )
        return row if isinstance(row, dict) else {}

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
