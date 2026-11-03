from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ProfileRole = Literal["admin", "decideur", "pmu", "analyste"]
WorkspaceRole = Literal["admin", "member", "viewer"]
# sent : lien envoyé par e-mail ; attached : le compte existait et était
# confirmé, il est rattaché sans e-mail.
Delivery = Literal["sent", "attached"]

EMAIL_PATTERN = r"^\s*[^@\s]+@[^@\s]+\.[^@\s]+\s*$"


class InvitationRequest(BaseModel):
    organization_id: UUID
    # Même contrôle que la base (organization_invitations_email_check).
    email: str = Field(max_length=320, pattern=EMAIL_PATTERN)
    full_name: str | None = Field(default=None, max_length=200)
    role: ProfileRole = "analyste"
    workspace_ids: list[UUID] = Field(default_factory=list)
    workspace_role: WorkspaceRole = "member"


class InvitationResult(BaseModel):
    id: UUID
    delivery: Delivery


class ProfileCreateRequest(BaseModel):
    """Compte créé à la main : l'administrateur choisit le mot de passe et le
    transmet lui-même, aucun e-mail ne part."""

    organization_id: UUID
    email: str = Field(max_length=320, pattern=EMAIL_PATTERN)
    full_name: str | None = Field(default=None, max_length=200)
    # 72 : au-delà, bcrypt (GoTrue) ignore la fin du mot de passe.
    password: str = Field(min_length=8, max_length=72)
    role: ProfileRole = "analyste"
    workspace_ids: list[UUID] = Field(default_factory=list)
    workspace_role: WorkspaceRole = "member"


class ProfileCreateResult(BaseModel):
    id: UUID
