from typing import Protocol

import httpx

from app.api.dependencies import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway
from app.modules.frameworks.schemas import (
    ApplicableCriterion,
    GeneratePillarsRequest,
    PillarGenerationContext,
    PromptDefinition,
    ReferencePillar,
)


class PillarContextRepository(Protocol):
    async def load(
        self, request: GeneratePillarsRequest, user: CurrentUser
    ) -> PillarGenerationContext: ...

    async def mark_running(self, request: GeneratePillarsRequest) -> None: ...

    async def persist(
        self,
        request: GeneratePillarsRequest,
        approach_id: str,
        result: dict,
    ) -> None: ...

    async def mark_failed(
        self,
        request: GeneratePillarsRequest,
        message: str,
    ) -> None: ...


class SupabasePillarContextRepository:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._gateway = SupabaseGateway(settings, client)

    async def load(
        self, request: GeneratePillarsRequest, user: CurrentUser
    ) -> PillarGenerationContext:
        workspaces = await self._gateway.rest_get(
            "workspaces",
            token=user.token,
            params={
                "id": f"eq.{request.workspace_id}",
                "select": (
                    "kind,target_country,financiers,themes,declared_stage,"
                    "start_year,end_year"
                ),
                "limit": "1",
            },
        )
        if not workspaces:
            raise ApiError(
                status_code=404,
                code="WORKSPACE_NOT_FOUND",
                detail="Workspace introuvable",
                context={"workspace_id": str(request.workspace_id)},
            )

        approach_id = request.approach_id
        if approach_id is None and request.job_id is not None:
            jobs = await self._gateway.rest_get(
                "pillar_generation_jobs",
                token=user.token,
                params={
                    "id": f"eq.{request.job_id}",
                    "workspace_id": f"eq.{request.workspace_id}",
                    "select": "approach_id",
                    "limit": "1",
                },
            )
            if jobs and jobs[0].get("approach_id"):
                approach_id = jobs[0]["approach_id"]

        approach_params = {
            "workspace_id": f"eq.{request.workspace_id}",
            "status": "eq.confirmed",
            "select": (
                "id,status,cycle,instrument_module,instrument_subtype,"
                "complexity_class,locale,ref_frameworks(code)"
            ),
            "order": "version.desc",
            "limit": "1",
        }
        if approach_id is not None:
            approach_params["id"] = f"eq.{approach_id}"

        approaches = await self._gateway.rest_get(
            "approaches",
            token=user.token,
            params=approach_params,
        )
        if not approaches:
            raise ApiError(
                status_code=404,
                code="APPROACH_NOT_FOUND",
                detail="Approche d'évaluation introuvable",
                context={"approach_id": str(approach_id) if approach_id else None},
            )
        approach = approaches[0]
        if approach.get("status") != "confirmed":
            raise ApiError(
                status_code=409,
                code="APPROACH_NOT_CONFIRMED",
                detail="L'approche doit être confirmée avant la génération",
                context={"approach_id": str(approach["id"])},
            )

        criteria_rows = await self._gateway.rest_get(
            "approach_criteria",
            token=user.token,
            params={
                "approach_id": f"eq.{approach['id']}",
                "applicability": "neq.non_applicable",
                "select": "applicability,weight,ref_criteria(code)",
            },
        )
        reference_rows = await self._gateway.rpc(
            "get_localized_ref_pillars",
            token=user.token,
            payload={
                "requested_module": approach["instrument_module"],
                "requested_cycle": approach["cycle"],
                "requested_locale": approach.get("locale", "fr"),
            },
        )
        prompt_data = await self._gateway.internal_rpc(
            "get_prompt_definition",
            payload={
                "prompt_code": "generate_pillars",
                "prompt_version": "v1.0",
            },
        )

        workspace = workspaces[0]
        framework = approach.get("ref_frameworks") or {}
        return PillarGenerationContext(
            approach_id=approach["id"],
            object_type=workspace["kind"],
            country=workspace.get("target_country") or "",
            financiers=workspace.get("financiers") or [],
            themes=workspace.get("themes") or [],
            stage=workspace.get("declared_stage") or "implementation",
            start_year=workspace.get("start_year") or 0,
            end_year=workspace.get("end_year") or 0,
            cycle=approach["cycle"],
            instrument_module=approach["instrument_module"],
            instrument_subtype=approach["instrument_subtype"],
            complexity_class=approach["complexity_class"],
            framework_code=framework.get("code", "themiros"),
            applicable_criteria=[
                ApplicableCriterion(
                    code=row["ref_criteria"]["code"],
                    applicability=row["applicability"],
                    weight=row["weight"],
                )
                for row in criteria_rows
            ],
            ref_pillars=[ReferencePillar.model_validate(row) for row in reference_rows],
            prompt=PromptDefinition.model_validate(prompt_data),
        )

    async def mark_running(self, request: GeneratePillarsRequest) -> None:
        if request.job_id is None:
            return
        await self._gateway.internal_rpc(
            "start_pillar_generation",
            payload={"target_job_id": str(request.job_id)},
        )

    async def persist(
        self,
        request: GeneratePillarsRequest,
        approach_id: str,
        result: dict,
    ) -> None:
        await self._gateway.internal_rpc(
            "persist_generated_pillars",
            payload={
                "target_workspace_id": str(request.workspace_id),
                "target_approach_id": str(approach_id),
                "generation_result": result,
                "target_job_id": str(request.job_id) if request.job_id else None,
            },
        )

    async def mark_failed(
        self,
        request: GeneratePillarsRequest,
        message: str,
    ) -> None:
        if request.job_id is None:
            return
        await self._gateway.internal_rpc(
            "fail_pillar_generation",
            payload={
                "target_job_id": str(request.job_id),
                "failure_message": message,
            },
        )
