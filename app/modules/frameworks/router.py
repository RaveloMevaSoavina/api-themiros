from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Request

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.modules.frameworks.generator import PillarGenerator, get_pillar_generator
from app.modules.frameworks.repository import (
    PillarContextRepository,
    SupabasePillarContextRepository,
)
from app.modules.frameworks.schemas import (
    GeneratePillarsRequest,
    GeneratePillarsResponse,
)
from app.modules.frameworks.service import PillarGenerationService

router = APIRouter(prefix="/frameworks", tags=["frameworks"])


def get_generation_service(
    request: Request,
    generator: Annotated[PillarGenerator, Depends(get_pillar_generator)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PillarGenerationService:
    client: httpx.AsyncClient = request.app.state.http_client
    repository: PillarContextRepository = SupabasePillarContextRepository(
        settings, client
    )
    return PillarGenerationService(generator, repository)


@router.post(
    "/generate-pillars",
    response_model=GeneratePillarsResponse,
    responses={
        401: {"description": "Missing or invalid Supabase JWT"},
        409: {"description": "Approach not confirmed"},
        503: {"description": "AI provider unavailable"},
    },
)
async def generate_pillars(
    payload: GeneratePillarsRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[PillarGenerationService, Depends(get_generation_service)],
) -> GeneratePillarsResponse:
    return await service.generate(payload, user)
