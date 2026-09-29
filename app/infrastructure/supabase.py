from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ApiError


class SupabaseGateway:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(
            self._settings.supabase_url and self._settings.supabase_service_role_key
        )

    async def ping(self) -> tuple[bool, str]:
        if not self.configured:
            return False, "not_configured"

        response = await self._client.get(
            f"{self._settings.supabase_url.rstrip('/')}/rest/v1/",
            headers={
                "apikey": self._settings.supabase_service_role_key or "",
                "Authorization": (
                    f"Bearer {self._settings.supabase_service_role_key or ''}"
                ),
            },
            timeout=3,
        )
        if response.is_success:
            return True, "ok"
        return False, f"http_{response.status_code}"

    async def rest_get(
        self,
        path: str,
        *,
        token: str,
        params: dict[str, str] | None = None,
    ) -> list[dict]:
        response = await self._client.get(
            self._rest_url(path),
            params=params,
            headers=self._user_headers(token),
            timeout=10,
        )
        return self._json_rows(response)

    async def rpc(
        self,
        function_name: str,
        *,
        token: str,
        payload: dict,
    ) -> list[dict]:
        response = await self._client.post(
            self._rest_url(f"rpc/{function_name}"),
            json=payload,
            headers=self._user_headers(token),
            timeout=10,
        )
        return self._json_rows(response)

    async def internal_rpc(
        self,
        function_name: str,
        *,
        payload: dict,
    ) -> Any:
        if not self._settings.supabase_service_role_key:
            raise ApiError(
                status_code=503,
                code="DATABASE_NOT_CONFIGURED",
                detail="La clé serveur Supabase n'est pas configurée",
            )
        response = await self._client.post(
            self._rest_url(f"rpc/{function_name}"),
            json=payload,
            headers={
                "apikey": self._settings.supabase_service_role_key,
                "Authorization": (f"Bearer {self._settings.supabase_service_role_key}"),
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if not response.is_success:
            raise ApiError(
                status_code=503,
                code="DATABASE_WRITE_FAILED",
                detail="Supabase n'a pas pu enregistrer le cadre généré",
                context={"upstream_status": response.status_code},
            )
        return response.json() if response.content else None

    def _rest_url(self, path: str) -> str:
        if not self._settings.supabase_url:
            raise ApiError(
                status_code=503,
                code="DATABASE_NOT_CONFIGURED",
                detail="Supabase n'est pas configuré côté API",
            )
        return f"{self._settings.supabase_url.rstrip('/')}/rest/v1/{path.lstrip('/')}"

    def _user_headers(self, token: str) -> dict[str, str]:
        if not self._settings.supabase_service_role_key:
            raise ApiError(
                status_code=503,
                code="DATABASE_NOT_CONFIGURED",
                detail="La clé serveur Supabase n'est pas configurée",
            )
        return {
            "apikey": self._settings.supabase_service_role_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _json_rows(response: httpx.Response) -> list[dict]:
        if response.status_code in {401, 403}:
            raise ApiError(
                status_code=response.status_code,
                code="DATABASE_ACCESS_DENIED",
                detail="Accès aux données du workspace refusé",
            )
        if not response.is_success:
            raise ApiError(
                status_code=503,
                code="DATABASE_UNAVAILABLE",
                detail="Supabase n'a pas pu fournir le contexte de génération",
                context={"upstream_status": response.status_code},
            )
        data = response.json()
        if not isinstance(data, list):
            raise ApiError(
                status_code=502,
                code="DATABASE_RESPONSE_INVALID",
                detail="Réponse Supabase inattendue",
            )
        return data
