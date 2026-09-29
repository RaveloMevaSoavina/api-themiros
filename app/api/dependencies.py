from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import httpx
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.errors import ApiError

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class CurrentUser:
    id: UUID
    email: str | None
    token: str


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiError(
            status_code=401,
            code="AUTHENTICATION_REQUIRED",
            detail="Authentification requise",
        )
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise ApiError(
            status_code=503,
            code="AUTHENTICATION_NOT_CONFIGURED",
            detail="La validation des jetons Supabase n'est pas configurée",
        )

    token = credentials.credentials
    try:
        response = await request.app.state.http_client.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={
                "apikey": settings.supabase_service_role_key,
                "Authorization": f"Bearer {token}",
            },
            timeout=5,
        )
    except httpx.HTTPError as error:
        raise ApiError(
            status_code=503,
            code="AUTHENTICATION_SERVICE_UNAVAILABLE",
            detail="Le service d'authentification Supabase est indisponible",
        ) from error

    if response.status_code in {401, 403}:
        raise ApiError(
            status_code=401,
            code="INVALID_ACCESS_TOKEN",
            detail="Jeton d'accès invalide ou expiré",
        )
    if not response.is_success:
        raise ApiError(
            status_code=503,
            code="AUTHENTICATION_SERVICE_UNAVAILABLE",
            detail="Supabase n'a pas pu valider le jeton d'accès",
            context={"upstream_status": response.status_code},
        )

    try:
        user = response.json()
        user_id = UUID(user["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError(
            status_code=502,
            code="AUTHENTICATION_RESPONSE_INVALID",
            detail="La réponse du service d'authentification est invalide",
        ) from error

    return CurrentUser(id=user_id, email=user.get("email"), token=token)
