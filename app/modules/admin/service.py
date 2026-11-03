"""Invitations du backoffice.

Ordre : l'invitation est d'abord enregistrée (la base refuse une adresse
déjà invitée ou rattachée à une autre organisation), puis GoTrue envoie le
lien en créant le compte, puis le compte est rattaché à l'organisation et
aux espaces choisis. Si l'envoi échoue, l'invitation enregistrée est
retirée : le backoffice ne montre pas une invitation que personne n'a reçue.
"""

import logging
from contextlib import suppress
from uuid import UUID

from app.api.dependencies import CurrentUser
from app.core.errors import ApiError
from app.modules.admin.repository import AdminRepository
from app.modules.admin.schemas import Delivery, InvitationRequest, InvitationResult

logger = logging.getLogger(__name__)


class InvitationService:
    def __init__(
        self, repository: AdminRepository, *, redirect_to: str | None = None
    ) -> None:
        self._repository = repository
        self._redirect_to = redirect_to

    async def invite(
        self, request: InvitationRequest, user: CurrentUser
    ) -> InvitationResult:
        await self._require_platform_admin(user)
        created = await self._repository.create_invitation(request, token=user.token)
        invitation_id = UUID(created["id"])
        try:
            delivery = await self._deliver(
                invitation_id,
                created["email"],
                created.get("full_name"),
                confirmed=bool(created["confirmed"]),
                existing=bool(created["existing_user"]),
            )
        except ApiError:
            with suppress(ApiError):
                await self._repository.discard_invitation(invitation_id)
            raise
        return InvitationResult(id=invitation_id, delivery=delivery)

    async def resend(self, invitation_id: UUID, user: CurrentUser) -> InvitationResult:
        await self._require_platform_admin(user)
        invitation = await self._repository.invitation(invitation_id, token=user.token)
        if invitation["status"] != "pending":
            raise ApiError(
                status_code=409,
                code="INVITATION_NOT_PENDING",
                detail="Seule une invitation en attente se renvoie",
                context={"status": invitation["status"]},
            )
        delivery = await self._deliver(
            invitation_id,
            invitation["email"],
            invitation.get("full_name"),
            confirmed=bool(invitation["confirmed"]),
            existing=invitation.get("user_id") is not None,
        )
        return InvitationResult(id=invitation_id, delivery=delivery)

    async def revoke(self, invitation_id: UUID, user: CurrentUser) -> None:
        await self._require_platform_admin(user)
        revoked = await self._repository.revoke_invitation(
            invitation_id, token=user.token
        )
        # Le compte n'existait que par l'invitation et ne s'est jamais
        # connecté : il disparaît avec elle.
        if revoked.get("delete_user") and revoked.get("user_id"):
            await self._repository.delete_user(str(revoked["user_id"]))

    async def _deliver(
        self,
        invitation_id: UUID,
        email: str,
        full_name: str | None,
        *,
        confirmed: bool,
        existing: bool,
    ) -> Delivery:
        if not confirmed:
            sent = await self._repository.send_invite(
                email, full_name=full_name, redirect_to=self._redirect_to
            )
            if sent is not None:
                await self._repository.apply_invitation(
                    invitation_id, email_sent=True, user_created=not existing
                )
                return "sent"
            logger.info("invitation %s: account confirmed meanwhile", invitation_id)
        await self._repository.apply_invitation(
            invitation_id, email_sent=False, user_created=False
        )
        return "attached"

    async def _require_platform_admin(self, user: CurrentUser) -> None:
        if not await self._repository.is_platform_admin(token=user.token):
            raise ApiError(
                status_code=403,
                code="PLATFORM_ADMIN_REQUIRED",
                detail="Réservé aux administrateurs de la plateforme",
            )
