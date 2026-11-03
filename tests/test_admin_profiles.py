from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import CurrentUser, get_current_user
from app.core.errors import ApiError
from app.main import app
from app.modules.admin.router import get_profile_service
from app.modules.admin.schemas import ProfileCreateRequest
from app.modules.admin.service import ProfileService

ADMIN = CurrentUser(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    email="staff@evoranq.com",
    token="admin-token",
)
ORGANIZATION_ID = "33333333-3333-4333-8333-333333333333"
NEW_USER_ID = "44444444-4444-4444-8444-444444444444"
WORKSPACE_ID = "55555555-5555-4555-8555-555555555555"


def request(**overrides) -> ProfileCreateRequest:
    return ProfileCreateRequest.model_validate(
        {
            "organization_id": ORGANIZATION_ID,
            "email": " Awa@Example.org ",
            "full_name": "Awa Ndiaye",
            "password": "un-mot-de-passe",
            "role": "pmu",
            **overrides,
        }
    )


class FakeRepository:
    def __init__(
        self,
        *,
        admin: bool = True,
        email_taken: bool = False,
        member_fails: bool = False,
        ban_fails: bool = False,
        has_records: bool = False,
    ) -> None:
        self._admin = admin
        self._email_taken = email_taken
        self._member_fails = member_fails
        self._ban_fails = ban_fails
        self._has_records = has_records
        self.prepared: list[UUID] = []
        self.archived: list[UUID] = []
        self.restored: list[UUID] = []
        self.bans: list[tuple[UUID, bool]] = []
        self.created: list[dict] = []
        self.profiles: list[dict] = []
        self.members: list[tuple[UUID, str]] = []
        self.deleted_users: list[str] = []

    async def is_platform_admin(self, *, token):
        assert token == ADMIN.token
        return self._admin

    async def create_user(self, email, *, password, full_name):
        if self._email_taken:
            raise ApiError(status_code=409, code="EMAIL_ALREADY_REGISTERED", detail="x")
        self.created.append(
            {"email": email, "password": password, "full_name": full_name}
        )
        return NEW_USER_ID

    async def update_profile(
        self, profile_id, *, full_name, role, organization_id, token
    ):
        self.profiles.append(
            {
                "id": profile_id,
                "full_name": full_name,
                "role": role,
                "organization_id": str(organization_id),
            }
        )

    async def set_workspace_member(self, workspace_id, profile_id, *, role, token):
        if self._member_fails:
            raise ApiError(
                status_code=409, code="MEMBER_NOT_IN_ORGANIZATION", detail="x"
            )
        self.members.append((workspace_id, role))

    async def delete_user(self, user_id):
        self.deleted_users.append(user_id)

    async def archive_profile(self, profile_id, *, token):
        self.archived.append(profile_id)

    async def restore_profile(self, profile_id, *, token):
        self.restored.append(profile_id)

    async def prepare_profile_deletion(self, profile_id, *, token):
        if self._has_records:
            raise ApiError(status_code=409, code="PROFILE_HAS_RECORDS", detail="x")
        self.prepared.append(profile_id)

    async def set_banned(self, user_id, *, banned):
        if self._ban_fails:
            raise ApiError(status_code=503, code="USER_BAN_FAILED", detail="x")
        self.bans.append((user_id, banned))


async def test_the_account_joins_the_organization_and_its_workspaces() -> None:
    repository = FakeRepository()

    result = await ProfileService(repository).create(
        request(workspace_ids=[WORKSPACE_ID, WORKSPACE_ID], workspace_role="viewer"),
        ADMIN,
    )

    assert result.id == UUID(NEW_USER_ID)
    assert repository.created == [
        {
            "email": "awa@example.org",
            "password": "un-mot-de-passe",
            "full_name": "Awa Ndiaye",
        }
    ]
    assert repository.profiles == [
        {
            "id": NEW_USER_ID,
            "full_name": "Awa Ndiaye",
            "role": "pmu",
            "organization_id": ORGANIZATION_ID,
        }
    ]
    assert repository.members == [(UUID(WORKSPACE_ID), "viewer")]


async def test_without_a_name_the_profile_takes_the_address() -> None:
    repository = FakeRepository()

    await ProfileService(repository).create(request(full_name="  "), ADMIN)

    assert repository.created[0]["full_name"] is None
    assert repository.profiles[0]["full_name"] == "awa"


async def test_a_failed_attachment_deletes_the_account() -> None:
    repository = FakeRepository(member_fails=True)

    with pytest.raises(ApiError) as error:
        await ProfileService(repository).create(
            request(workspace_ids=[WORKSPACE_ID]), ADMIN
        )

    assert error.value.code == "MEMBER_NOT_IN_ORGANIZATION"
    assert repository.deleted_users == [NEW_USER_ID]


async def test_a_taken_address_creates_nothing() -> None:
    repository = FakeRepository(email_taken=True)

    with pytest.raises(ApiError) as error:
        await ProfileService(repository).create(request(), ADMIN)

    assert error.value.code == "EMAIL_ALREADY_REGISTERED"
    assert repository.profiles == []
    assert repository.deleted_users == []


def post_profile(repository: FakeRepository, payload: dict):
    app.dependency_overrides[get_current_user] = lambda: ADMIN
    app.dependency_overrides[get_profile_service] = lambda: ProfileService(repository)
    try:
        with TestClient(app) as client:
            return client.post("/api/v1/admin/profiles", json=payload)
    finally:
        app.dependency_overrides.clear()


def test_the_endpoint_is_reserved_to_platform_admins() -> None:
    repository = FakeRepository(admin=False)

    response = post_profile(
        repository,
        {
            "organization_id": ORGANIZATION_ID,
            "email": "awa@example.org",
            "password": "un-mot-de-passe",
        },
    )

    assert response.status_code == 403
    assert response.json()["code"] == "PLATFORM_ADMIN_REQUIRED"
    assert repository.created == []


def test_a_short_password_is_rejected_before_anything_is_created() -> None:
    repository = FakeRepository()

    response = post_profile(
        repository,
        {
            "organization_id": ORGANIZATION_ID,
            "email": "awa@example.org",
            "password": "court",
        },
    )

    assert response.status_code == 422
    assert repository.created == []


async def test_archiving_bans_the_account() -> None:
    repository = FakeRepository()

    await ProfileService(repository).archive(UUID(NEW_USER_ID), ADMIN)

    assert repository.archived == [UUID(NEW_USER_ID)]
    assert repository.bans == [(UUID(NEW_USER_ID), True)]
    assert repository.restored == []


async def test_an_account_that_cannot_be_banned_is_not_shown_archived() -> None:
    repository = FakeRepository(ban_fails=True)

    with pytest.raises(ApiError) as error:
        await ProfileService(repository).archive(UUID(NEW_USER_ID), ADMIN)

    assert error.value.code == "USER_BAN_FAILED"
    assert repository.restored == [UUID(NEW_USER_ID)]


async def test_restoring_unbans_before_clearing_the_archive() -> None:
    repository = FakeRepository()

    await ProfileService(repository).restore(UUID(NEW_USER_ID), ADMIN)

    assert repository.bans == [(UUID(NEW_USER_ID), False)]
    assert repository.restored == [UUID(NEW_USER_ID)]


async def test_a_failed_unban_leaves_the_profile_archived() -> None:
    repository = FakeRepository(ban_fails=True)

    with pytest.raises(ApiError):
        await ProfileService(repository).restore(UUID(NEW_USER_ID), ADMIN)

    assert repository.restored == []


def test_archiving_is_reserved_to_platform_admins() -> None:
    repository = FakeRepository(admin=False)
    app.dependency_overrides[get_current_user] = lambda: ADMIN
    app.dependency_overrides[get_profile_service] = lambda: ProfileService(repository)
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/v1/admin/profiles/{NEW_USER_ID}/archive")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert repository.archived == []
    assert repository.bans == []


async def test_deleting_an_unused_account_removes_it() -> None:
    repository = FakeRepository()

    await ProfileService(repository).delete(UUID(NEW_USER_ID), ADMIN)

    assert repository.prepared == [UUID(NEW_USER_ID)]
    assert repository.deleted_users == [NEW_USER_ID]


async def test_an_account_with_records_is_not_deleted() -> None:
    repository = FakeRepository(has_records=True)

    with pytest.raises(ApiError) as error:
        await ProfileService(repository).delete(UUID(NEW_USER_ID), ADMIN)

    assert error.value.code == "PROFILE_HAS_RECORDS"
    assert repository.deleted_users == []


def test_deleting_is_reserved_to_platform_admins() -> None:
    repository = FakeRepository(admin=False)
    app.dependency_overrides[get_current_user] = lambda: ADMIN
    app.dependency_overrides[get_profile_service] = lambda: ProfileService(repository)
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/v1/admin/profiles/{NEW_USER_ID}/delete")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert repository.deleted_users == []
