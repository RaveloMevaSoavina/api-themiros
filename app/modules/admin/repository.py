"""Accès Supabase du backoffice.

Les fonctions backoffice_* s'appellent avec le jeton de l'administrateur :
elles vérifient elles-mêmes qu'il administre la plateforme. Seuls le
rattachement d'un compte invité, l'envoi de l'e-mail, la création et la
suppression d'un compte passent par la clé serveur.
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

    async def create_user(
        self, email: str, *, password: str, full_name: str | None
    ) -> str:
        data: dict[str, str] = {"lang": "fr"}
        if full_name:
            data["full_name"] = full_name
        created = await self._gateway.auth_create_user(
            email, password=password, data=data
        )
        return str(created["id"])

    async def update_profile(
        self,
        profile_id: str,
        *,
        full_name: str,
        role: str,
        organization_id: UUID,
        token: str,
    ) -> None:
        await self._gateway.rpc_value(
            "backoffice_update_profile",
            token=token,
            payload={
                "p_profile_id": profile_id,
                "p_full_name": full_name,
                "p_role": role,
                "p_organization_id": str(organization_id),
            },
        )

    async def archive_profile(self, profile_id: UUID, *, token: str) -> None:
        await self._gateway.rpc_value(
            "backoffice_archive_profile",
            token=token,
            payload={"p_profile_id": str(profile_id)},
        )

    async def restore_profile(self, profile_id: UUID, *, token: str) -> None:
        await self._gateway.rpc_value(
            "backoffice_restore_profile",
            token=token,
            payload={"p_profile_id": str(profile_id)},
        )

    async def set_banned(self, user_id: UUID, *, banned: bool) -> None:
        await self._gateway.auth_set_banned(str(user_id), banned=banned)

    async def set_workspace_member(
        self, workspace_id: UUID, profile_id: str, *, role: str, token: str
    ) -> None:
        await self._gateway.rpc_value(
            "backoffice_set_workspace_member",
            token=token,
            payload={
                "p_workspace_id": str(workspace_id),
                "p_profile_id": profile_id,
                "p_role": role,
            },
        )
