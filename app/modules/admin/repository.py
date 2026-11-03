"""Accès Supabase du backoffice.

Les fonctions backoffice_* s'appellent avec le jeton de l'administrateur :
elles vérifient elles-mêmes qu'il administre la plateforme. Seuls le
rattachement d'un compte invité et l'envoi de l'e-mail passent par la clé
serveur.
"""

from typing import Any
from uuid import UUID

from app.infrastructure.supabase import SupabaseGateway
from app.modules.admin.schemas import InvitationRequest


class AdminRepository:
    def __init__(self, gateway: SupabaseGateway) -> None:
        self._gateway = gateway

    async def is_platform_admin(self, *, token: str) -> bool:
        return bool(
            await self._gateway.rpc_value("is_platform_admin", token=token, payload={})
        )

    async def create_invitation(
        self, request: InvitationRequest, *, token: str
    ) -> dict[str, Any]:
        return await self._gateway.rpc_value(
            "backoffice_create_invitation",
            token=token,
            payload={
                "p_organization_id": str(request.organization_id),
                "p_email": request.email,
                "p_full_name": request.full_name,
                "p_role": request.role,
                "p_workspace_ids": [str(id_) for id_ in request.workspace_ids],
                "p_workspace_role": request.workspace_role,
            },
        )

    async def invitation(self, invitation_id: UUID, *, token: str) -> dict[str, Any]:
        return await self._gateway.rpc_value(
            "backoffice_get_invitation",
            token=token,
            payload={"p_invitation_id": str(invitation_id)},
        )

    async def revoke_invitation(
        self, invitation_id: UUID, *, token: str
    ) -> dict[str, Any]:
        return await self._gateway.rpc_value(
            "backoffice_revoke_invitation",
            token=token,
            payload={"p_invitation_id": str(invitation_id)},
        )

    async def apply_invitation(
        self, invitation_id: UUID, *, email_sent: bool, user_created: bool
    ) -> None:
        await self._gateway.service_rpc(
            "backoffice_apply_invitation",
            payload={
                "p_invitation_id": str(invitation_id),
                "p_email_sent": email_sent,
                "p_user_created": user_created,
            },
        )

    async def discard_invitation(self, invitation_id: UUID) -> None:
        await self._gateway.service_rpc(
            "backoffice_discard_invitation",
            payload={"p_invitation_id": str(invitation_id)},
        )

    async def send_invite(
        self, email: str, *, full_name: str | None, redirect_to: str | None
    ) -> dict | None:
        # full_name et lang alimentent handle_new_user, qui crée le profil.
        data: dict[str, str] = {"lang": "fr"}
        if full_name:
            data["full_name"] = full_name
        return await self._gateway.auth_invite(
            email, data=data, redirect_to=redirect_to
        )

    async def delete_user(self, user_id: str) -> None:
        await self._gateway.auth_delete_user(user_id)
