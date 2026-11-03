from typing import Annotated
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.admin.repository import AdminRepository
from app.modules.admin.schemas import InvitationRequest, InvitationResult
from app.modules.admin.service import InvitationService

router = APIRouter(prefix="/admin", tags=["admin"])


def get_invitation_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> InvitationService:
    client: httpx.AsyncClient = request.app.state.http_client
    return InvitationService(
        AdminRepository(SupabaseGateway(settings, client)),
        redirect_to=settings.invitation_redirect_url,
    )


@router.post(
    "/invitations",
    response_model=InvitationResult,
    status_code=201,
    responses={
        401: {"description": "Missing or invalid Supabase JWT"},
        403: {"description": "Not a platform administrator"},
        409: {"description": "Already invited or in another organization"},
    },
)
async def create_invitation(
    payload: InvitationRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[InvitationService, Depends(get_invitation_service)],
) -> InvitationResult:
    """Invite une personne dans une organisation et, au choix, ses espaces."""
    return await service.invite(payload, user)


@router.post("/invitations/{invitation_id}/resend", response_model=InvitationResult)
async def resend_invitation(
    invitation_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[InvitationService, Depends(get_invitation_service)],
) -> InvitationResult:
    return await service.resend(invitation_id, user)


@router.post("/invitations/{invitation_id}/revoke", status_code=204)
async def revoke_invitation(
    invitation_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[InvitationService, Depends(get_invitation_service)],
) -> Response:
    await service.revoke(invitation_id, user)
    return Response(status_code=204)
