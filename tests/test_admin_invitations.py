from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import CurrentUser, get_current_user
from app.core.errors import ApiError
from app.main import app
from app.modules.admin.router import get_invitation_service
from app.modules.admin.schemas import InvitationRequest
from app.modules.admin.service import InvitationService

ADMIN = CurrentUser(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    email="staff@evoranq.com",
    token="admin-token",
)
INVITATION_ID = "22222222-2222-4222-8222-222222222222"
ORGANIZATION_ID = "33333333-3333-4333-8333-333333333333"
NEW_USER_ID = "44444444-4444-4444-8444-444444444444"


def request(**overrides) -> InvitationRequest:
    return InvitationRequest.model_validate(
        {
            "organization_id": ORGANIZATION_ID,
            "email": "awa@example.org",
            "full_name": "Awa Ndiaye",
            "role": "pmu",
            **overrides,
        }
    )


class FakeRepository:
    def __init__(
        self,
        *,
        admin: bool = True,
        existing_user: bool = False,
        confirmed: bool = False,
        registered_meanwhile: bool = False,
        send_fails: bool = False,
        invitation: dict | None = None,
        revoked: dict | None = None,
    ) -> None:
        self._admin = admin
        self._existing_user = existing_user
        self._confirmed = confirmed
        self._registered_meanwhile = registered_meanwhile
        self._send_fails = send_fails
        self._invitation = invitation
        self._revoked = revoked or {"user_id": None, "delete_user": False}
        self.sent: list[tuple[str, str | None]] = []
        self.applied: list[dict] = []
        self.discarded: list[UUID] = []
        self.deleted_users: list[str] = []

    async def is_platform_admin(self, *, token):
        assert token == ADMIN.token
        return self._admin

    async def create_invitation(self, invitation_request, *, token):
        return {
            "id": INVITATION_ID,
            "email": invitation_request.email.strip().lower(),
            "full_name": invitation_request.full_name,
            "existing_user": self._existing_user,
            "confirmed": self._confirmed,
        }

    async def invitation(self, invitation_id, *, token):
        return self._invitation

    async def revoke_invitation(self, invitation_id, *, token):
        return self._revoked

    async def apply_invitation(self, invitation_id, *, email_sent, user_created):
        self.applied.append({"email_sent": email_sent, "user_created": user_created})

    async def discard_invitation(self, invitation_id):
        self.discarded.append(invitation_id)

    async def send_invite(self, email, *, full_name, redirect_to):
        if self._send_fails:
            raise ApiError(status_code=503, code="INVITATION_EMAIL_FAILED", detail="x")
        self.sent.append((email, redirect_to))
        return None if self._registered_meanwhile else {"id": NEW_USER_ID}

    async def delete_user(self, user_id):
        self.deleted_users.append(user_id)


async def test_a_new_address_receives_a_link_and_joins_the_organization() -> None:
    repository = FakeRepository()
    service = InvitationService(repository, redirect_to="https://app.themiros.com")

    result = await service.invite(request(), ADMIN)

    assert result.delivery == "sent"
    assert repository.sent == [("awa@example.org", "https://app.themiros.com")]
    assert repository.applied == [{"email_sent": True, "user_created": True}]


async def test_a_confirmed_account_is_attached_without_email() -> None:
    repository = FakeRepository(existing_user=True, confirmed=True)

    result = await InvitationService(repository).invite(request(), ADMIN)

    assert result.delivery == "attached"
    assert repository.sent == []
    assert repository.applied == [{"email_sent": False, "user_created": False}]


async def test_an_unconfirmed_account_gets_a_new_link_but_is_not_ours() -> None:
    repository = FakeRepository(existing_user=True, confirmed=False)

    await InvitationService(repository).invite(request(), ADMIN)

    assert repository.applied == [{"email_sent": True, "user_created": False}]


async def test_an_account_confirmed_meanwhile_is_attached() -> None:
    repository = FakeRepository(registered_meanwhile=True)

    result = await InvitationService(repository).invite(request(), ADMIN)

    assert result.delivery == "attached"
    assert repository.applied == [{"email_sent": False, "user_created": False}]


async def test_a_failed_email_leaves_no_invitation_behind() -> None:
    repository = FakeRepository(send_fails=True)

    with pytest.raises(ApiError) as error:
        await InvitationService(repository).invite(request(), ADMIN)

    assert error.value.code == "INVITATION_EMAIL_FAILED"
    assert repository.discarded == [UUID(INVITATION_ID)]
    assert repository.applied == []


async def test_only_pending_invitations_are_resent() -> None:
    repository = FakeRepository(
        invitation={
            "id": INVITATION_ID,
            "email": "awa@example.org",
            "status": "accepted",
            "confirmed": True,
            "user_id": NEW_USER_ID,
        }
    )

    with pytest.raises(ApiError) as error:
        await InvitationService(repository).resend(UUID(INVITATION_ID), ADMIN)

    assert error.value.code == "INVITATION_NOT_PENDING"
    assert repository.sent == []


async def test_revoking_deletes_the_account_the_invitation_created() -> None:
    repository = FakeRepository(revoked={"user_id": NEW_USER_ID, "delete_user": True})

    await InvitationService(repository).revoke(UUID(INVITATION_ID), ADMIN)

    assert repository.deleted_users == [NEW_USER_ID]


async def test_revoking_keeps_an_account_that_existed_before() -> None:
    repository = FakeRepository(revoked={"user_id": NEW_USER_ID, "delete_user": False})

    await InvitationService(repository).revoke(UUID(INVITATION_ID), ADMIN)

    assert repository.deleted_users == []


def test_the_endpoint_is_reserved_to_platform_admins() -> None:
    repository = FakeRepository(admin=False)
    app.dependency_overrides[get_current_user] = lambda: ADMIN
    app.dependency_overrides[get_invitation_service] = lambda: InvitationService(
        repository
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/admin/invitations",
                json={"organization_id": ORGANIZATION_ID, "email": "awa@example.org"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert response.json()["code"] == "PLATFORM_ADMIN_REQUIRED"
    assert repository.sent == []


def test_an_invalid_address_is_rejected_before_anything_is_stored() -> None:
    app.dependency_overrides[get_current_user] = lambda: ADMIN
    app.dependency_overrides[get_invitation_service] = lambda: InvitationService(
        FakeRepository()
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/admin/invitations",
                json={"organization_id": ORGANIZATION_ID, "email": "pas-une-adresse"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
