from typing import Annotated
from urllib.parse import quote
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
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


MAX_SPANS = 20


def parse_spans(value: str | None) -> list[tuple[int, int]] | None:
    """« 0-154,155-300 » : positions des phrases citées dans l'extrait."""
    if not value:
        return None
    spans: list[tuple[int, int]] = []
    for part in value.split(","):
        start, separator, end = part.partition("-")
        if not separator or not start.isdigit() or not end.isdigit():
            raise ApiError(
                status_code=422,
                code="EVIDENCE_SPANS_INVALID",
                detail="Positions de passage invalides",
            )
        spans.append((int(start), int(end)))
    if len(spans) > MAX_SPANS:
        raise ApiError(
            status_code=422,
            code="EVIDENCE_SPANS_INVALID",
            detail="Trop de passages demandés",
        )
    return spans


@router.get(
    "/{evidence_id}/document",
    response_class=Response,
    responses={
        200: {"description": "PDF source, le passage cité surligné"},
        401: {"description": "Missing or invalid Supabase JWT"},
        404: {"description": "Evidence or document not found or not visible"},
        409: {"description": "Source document is not a PDF"},
        422: {"description": "Invalid spans"},
    },
)
async def evidence_document(
    evidence_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[EvidenceService, Depends(get_evidence_service)],
    spans: Annotated[str | None, Query(max_length=400)] = None,
) -> Response:
    """Document source d'une preuve. `spans` (« 0-154,155-300 ») restreint
    le surlignage aux phrases citées. `X-Evidence-Page` donne la page où
    s'ouvrir, `X-Evidence-Highlighted` dit si le passage a été retrouvé."""
    document = await service.document(evidence_id, user, parse_spans(spans))
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
