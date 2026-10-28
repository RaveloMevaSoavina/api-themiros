from typing import Annotated
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.exports.repository import ExportRepository
from app.modules.exports.schemas import ExportFormat, ExportRequest
from app.modules.exports.service import ExportService

router = APIRouter(prefix="/exports", tags=["exports"])


def get_export_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> ExportService:
    client: httpx.AsyncClient = request.app.state.http_client
    return ExportService(ExportRepository(SupabaseGateway(settings, client)))


@router.post(
    "/{run_id}/{export_format}",
    response_class=Response,
    responses={
        200: {"description": "Fichier du rapport"},
        401: {"description": "Missing or invalid Supabase JWT"},
        404: {"description": "Run not found or not visible"},
        409: {"description": "Run not completed"},
    },
)
async def export_run(
    run_id: UUID,
    export_format: ExportFormat,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[ExportService, Depends(get_export_service)],
    payload: ExportRequest | None = None,
) -> Response:
    """Rapport d'un run en PDF, DOCX, XLSX ou CSV (catalogue §9)."""
    exported = await service.export(
        run_id, export_format, payload or ExportRequest(), user
    )
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={"Content-Disposition": f'attachment; filename="{exported.filename}"'},
    )
