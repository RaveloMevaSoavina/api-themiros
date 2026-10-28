"""Lecture d'un run pour l'export, au nom de l'utilisateur.

Toutes les lectures passent par PostgREST avec le jeton de l'utilisateur :
la RLS des tables de résultats décide seule de ce qu'il peut exporter. Les
sélections reprennent celles des écrans du dashboard
(`evaluation-service.ts`).
"""

import asyncio
from typing import Any
from uuid import UUID

from app.infrastructure.supabase import SupabaseGateway
from app.modules.exports.report import RunRows

PILLAR_TEXT = (
    "name,description,source_locale,pillar_translations(locale,name,description)"
)
VARIABLE_TEXT = (
    "code,label,description,source_locale,"
    "pillar_variable_translations(locale,label,description)"
)


class ExportRepository:
    def __init__(self, gateway: SupabaseGateway) -> None:
        self._gateway = gateway

    async def run(self, run_id: UUID, *, token: str) -> dict[str, Any] | None:
        rows = await self._gateway.rest_get(
            "runs",
            token=token,
            params={
                "select": (
                    "id,workspace_id,status,engine_version,corpus_size,"
                    "corpus_seuil,created_at,completed_at,validated_at,"
                    "validated_by_name,workspaces(name)"
                ),
                "id": f"eq.{run_id}",
            },
        )
        return rows[0] if rows else None

    async def rows(self, run: dict[str, Any], *, token: str) -> RunRows:
        run_filter = f"eq.{run['id']}"

        def select(table: str, columns: str, **params: str):
            return self._gateway.rest_get(
                table,
                token=token,
                params={"select": columns, "run_id": run_filter, **params},
            )

        (
            steps,
            scores,
            variables,
            notes,
            alerts,
            evidences,
            syntheses,
            items,
            actions,
            corpus_changed,
        ) = await asyncio.gather(
            select("run_steps", "step_name,status"),
            select(
                "layer_a_scores",
                "id,pillar_id,score,delta,evolution_label,corpus_confidence,"
                f"documents_count,pillars({PILLAR_TEXT}),"
                "program_versions(label,order_index)",
            ),
            select(
                "intermediate_variables",
                "id,pillar_id,variable_code,value,state,documents_count,"
                "corpus_confidence,program_versions(label,order_index),"
                f"pillars(pillar_variables({VARIABLE_TEXT}))",
            ),
            select(
                "layer_b_notes",
                "id,criterion_id,note,status,documented_sub_questions,"
                "total_sub_questions,justification,ambiguity_flag,"
                "criteria(code,name)",
            ),
            select(
                "layer_c_alerts",
                "id,rule_code,alert_type,severity,message,qualified,status,"
                f"instruction_comment,pillar_id,criterion_id,pillars({PILLAR_TEXT}),"
                "criteria(name),alert_evidences(evidence_id)",
            ),
            select(
                "evidences",
                "id,document_id,page_number,section_title,excerpt,pillar_id,"
                "criterion_id,variable_code,similarity,"
                "documents(original_filename)",
                order="created_at,id",
            ),
            select("syntheses", "conclusion,limits,status"),
            select(
                "synthesis_items",
                "id,section,position,text,pillar_id,criterion_id,alert_id,"
                "synthesis_item_evidences(evidence_id)",
            ),
            select(
                "run_actions",
                "id,position,origin_type,pillar_id,criterion_id,alert_id,"
                "run_action_projections(persona,text,justification),"
                "run_action_evidences(evidence_id)",
            ),
            self._gateway.rpc_value(
                "run_corpus_changed", token=token, payload={"p_run_id": run["id"]}
            ),
        )
        return RunRows(
            run=run,
            steps=steps,
            scores=scores,
            variables=variables,
            notes=notes,
            alerts=alerts,
            evidences=evidences,
            synthesis=syntheses[0] if syntheses else None,
            items=items,
            actions=actions,
            corpus_changed=bool(corpus_changed),
        )

    async def record(
        self,
        run: dict[str, Any],
        *,
        token: str,
        user_id: UUID,
        operation: str,
        metadata: dict[str, Any],
    ) -> None:
        """Journalise l'export (format, run, auteur, horodatage) ; la RLS
        n'accepte qu'un membre de l'espace agissant en son nom."""
        await self._gateway.rest_insert(
            "audit_events",
            token=token,
            payload={
                "workspace_id": run["workspace_id"],
                "run_id": run["id"],
                "actor_type": "human",
                "actor_user_id": str(user_id),
                "operation": operation,
                "object_type": "run",
                "object_id": run["id"],
                "metadata": metadata,
            },
        )
