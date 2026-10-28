from typing import Annotated
from urllib.parse import quote
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.evidences.repository import EvidenceRepository
from app.modules.evidences.service import PDF_MEDIA_TYPE, EvidenceService

router = APIRouter(prefix="/evidences", tags=["evidences"])


def get_evidence_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> EvidenceService:
    client: httpx.AsyncClient = request.app.state.http_client
    return EvidenceService(
        EvidenceRepository(SupabaseGateway(settings, client), settings)
    )


@router.get(
    "/{evidence_id}/document",
    response_class=Response,
    responses={
        200: {"description": "PDF source, le passage cité surligné"},
        401: {"description": "Missing or invalid Supabase JWT"},
        404: {"description": "Evidence or document not found or not visible"},
        409: {"description": "Source document is not a PDF"},
    },
)
async def evidence_document(
    evidence_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[EvidenceService, Depends(get_evidence_service)],
) -> Response:
    """Document source d'une preuve. `X-Evidence-Page` donne la page où
    s'ouvrir, `X-Evidence-Highlighted` dit si le passage a été retrouvé."""
    document = await service.document(evidence_id, user)
    headers = {
        # Nom d'origine, souvent accentué : seul `filename*` le transporte.
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(document.filename)}",
        "Cache-Control": "private, no-store",
        "X-Evidence-Highlighted": "true" if document.highlighted else "false",
    }
    if document.page is not None:
        headers["X-Evidence-Page"] = str(document.page)
    return Response(
        content=document.content, media_type=PDF_MEDIA_TYPE, headers=headers
    )
