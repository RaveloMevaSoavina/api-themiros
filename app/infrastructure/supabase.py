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

    async def service_select(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> list[dict]:
        """Lecture avec la clé serveur : réservée aux traitements du moteur."""
        response = await self._client.get(
            self._rest_url(path),
            params=params,
            headers=self._service_headers(),
            timeout=15,
        )
        return self._json_rows(response)

    async def service_upsert(
        self, path: str, *, payload: dict | list[dict], on_conflict: str
    ) -> None:
        headers = self._service_headers()
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
        response = await self._client.post(
            self._rest_url(path),
            params={"on_conflict": on_conflict},
            json=payload,
            headers=headers,
            timeout=15,
        )
        if not response.is_success:
            raise self._service_error(response)

    async def service_rpc(
        self, function_name: str, *, payload: dict, request_timeout: float = 30
    ) -> Any:
        """Appel RPC serveur. Les erreurs métier des fonctions SQL
        (`raise exception using message = 'CODE'`) gardent leur code."""
        response = await self._client.post(
            self._rest_url(f"rpc/{function_name}"),
            json=payload,
            headers=self._service_headers(),
            timeout=request_timeout,
        )
        if not response.is_success:
            raise self._service_error(response)
        return response.json() if response.content else None

    async def storage_download(self, bucket: str, path: str) -> bytes:
        if not self._settings.supabase_url:
            raise ApiError(
                status_code=503,
                code="DATABASE_NOT_CONFIGURED",
                detail="Supabase n'est pas configuré côté API",
            )
        response = await self._client.get(
            f"{self._settings.supabase_url.rstrip('/')}"
            f"/storage/v1/object/{bucket}/{path.lstrip('/')}",
            headers=self._service_headers(),
            timeout=60,
        )
        if response.status_code in {400, 404}:
            raise ApiError(
                status_code=404,
                code="DOCUMENT_FILE_NOT_FOUND",
                detail="Le fichier du document est introuvable dans le stockage",
            )
        if not response.is_success:
            raise ApiError(
                status_code=503,
                code="DOCUMENT_DOWNLOAD_FAILED",
                detail="Le fichier n'a pas pu être téléchargé depuis le stockage",
                context={"upstream_status": response.status_code},
            )
        return response.content

    def _service_headers(self) -> dict[str, str]:
        if not self._settings.supabase_service_role_key:
            raise ApiError(
                status_code=503,
                code="DATABASE_NOT_CONFIGURED",
                detail="La clé serveur Supabase n'est pas configurée",
            )
        return {
            "apikey": self._settings.supabase_service_role_key,
            "Authorization": f"Bearer {self._settings.supabase_service_role_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _service_error(response: httpx.Response) -> ApiError:
        message = ""
        try:
            body = response.json()
            message = str(body.get("message") or "") if isinstance(body, dict) else ""
        except ValueError:
            pass
        if response.status_code < 500 and message.isupper() and " " not in message:
            return ApiError(
                status_code=409,
                code=message,
                detail=message,
                context={"upstream_status": response.status_code},
            )
        return ApiError(
            status_code=503,
            code="DATABASE_UNAVAILABLE",
            detail="Supabase n'a pas pu traiter la demande d'ingestion",
            context={"upstream_status": response.status_code},
        )

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
