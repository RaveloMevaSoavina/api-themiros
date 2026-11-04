import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ApiError

logger = logging.getLogger(__name__)


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

    async def rpc_value(
        self,
        function_name: str,
        *,
        token: str,
        payload: dict,
    ) -> Any:
        """Appel RPC au nom de l'utilisateur, pour une fonction scalaire."""
        response = await self._client.post(
            self._rest_url(f"rpc/{function_name}"),
            json=payload,
            headers=self._user_headers(token),
            timeout=10,
        )
        if response.status_code in {401, 403}:
            raise ApiError(
                status_code=response.status_code,
                code="DATABASE_ACCESS_DENIED",
                detail="Accès aux données du workspace refusé",
            )
        if not response.is_success:
            raise self._service_error(response)
        return response.json() if response.content else None

    async def rest_insert(self, path: str, *, token: str, payload: dict) -> None:
        """Insertion au nom de l'utilisateur : la RLS de la table s'applique."""
        headers = self._user_headers(token)
        headers["Prefer"] = "return=minimal"
        response = await self._client.post(
            self._rest_url(path), json=payload, headers=headers, timeout=10
        )
        if response.status_code in {401, 403}:
            raise ApiError(
                status_code=response.status_code,
                code="DATABASE_ACCESS_DENIED",
                detail="Accès aux données du workspace refusé",
            )
        if not response.is_success:
            raise self._service_error(response)

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

    async def service_insert(self, path: str, *, payload: dict | list[dict]) -> None:
        headers = self._service_headers()
        headers["Prefer"] = "return=minimal"
        response = await self._client.post(
            self._rest_url(path), json=payload, headers=headers, timeout=15
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

    async def auth_invite(
        self, email: str, *, data: dict, redirect_to: str | None
    ) -> dict | None:
        """Invitation GoTrue : crée le compte s'il n'existe pas et envoie le
        lien. None quand un compte confirmé porte déjà cette adresse."""
        response = await self._client.post(
            self._auth_url("invite"),
            params={"redirect_to": redirect_to} if redirect_to else None,
            json={"email": email, "data": data},
            headers=self._service_headers(),
            timeout=15,
        )
        if response.status_code == 422 and self._email_exists(response):
            return None
        if not response.is_success:
            raise self._auth_failure(
                response,
                code="INVITATION_EMAIL_FAILED",
                detail="Supabase n'a pas pu envoyer l'e-mail d'invitation",
            )
        return response.json()

    async def auth_create_user(self, email: str, *, password: str, data: dict) -> dict:
        """Compte créé par un administrateur : adresse confirmée d'office, aucun
        e-mail envoyé."""
        response = await self._client.post(
            self._auth_url("admin/users"),
            json={
                "email": email,
                "password": password,
                "email_confirm": True,
                "user_metadata": data,
            },
            headers=self._service_headers(),
            timeout=15,
        )
        if response.status_code == 422 and self._email_exists(response):
            raise ApiError(
                status_code=409,
                code="EMAIL_ALREADY_REGISTERED",
                detail="Un compte existe déjà avec cette adresse",
            )
        if response.status_code == 422 and self._error_code(response) == (
            "weak_password"
        ):
            raise ApiError(
                status_code=422,
                code="WEAK_PASSWORD",
                detail="Mot de passe refusé par Supabase Auth : trop faible",
            )
        if not response.is_success:
            raise self._auth_failure(
                response,
                code="USER_CREATE_FAILED",
                detail="Supabase n'a pas pu créer le compte",
            )
        return response.json()

    async def auth_set_banned(self, user_id: str, *, banned: bool) -> None:
        """Bannir empêche la connexion et le renouvellement de session ; le
        jeton déjà délivré reste valable jusqu'à son expiration."""
        response = await self._client.put(
            self._auth_url(f"admin/users/{user_id}"),
            # GoTrue n'a pas de bannissement sans fin : 100 ans.
            json={"ban_duration": "876000h" if banned else "none"},
            headers=self._service_headers(),
            timeout=15,
        )
        if not response.is_success:
            raise self._auth_failure(
                response,
                code="USER_BAN_FAILED",
                detail="Supabase n'a pas pu bloquer ou débloquer le compte",
            )

    async def auth_delete_user(self, user_id: str) -> None:
        response = await self._client.delete(
            self._auth_url(f"admin/users/{user_id}"),
            headers=self._service_headers(),
            timeout=15,
        )
        if response.status_code == 404:
            return
        if not response.is_success:
            raise self._auth_failure(
                response,
                code="USER_DELETE_FAILED",
                detail="Supabase n'a pas pu supprimer le compte",
            )

    @classmethod
    def _auth_failure(
        cls, response: httpx.Response, *, code: str, detail: str
    ) -> ApiError:
        """Échec GoTrue : la raison donnée par Supabase (SMTP, quota
        d'e-mails, ...) est journalisée et renvoyée au backoffice."""
        body = cls._auth_error_body(response)
        upstream_message = str(
            body.get("msg")
            or body.get("message")
            or body.get("error_description")
            or ""
        )[:300]
        logger.warning(
            "GoTrue %s: %s %s %s",
            code,
            response.status_code,
            body.get("error_code") or "",
            upstream_message,
        )
        return ApiError(
            status_code=503,
            code=code,
            detail=detail,
            context={
                "upstream_status": response.status_code,
                "upstream_code": str(body.get("error_code") or ""),
                "upstream_message": upstream_message,
            },
        )

    @staticmethod
    def _auth_error_body(response: httpx.Response) -> dict:
        try:
            body = response.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    @classmethod
    def _error_code(cls, response: httpx.Response) -> str:
        return str(cls._auth_error_body(response).get("error_code") or "")

    @classmethod
    def _email_exists(cls, response: httpx.Response) -> bool:
        body = cls._auth_error_body(response)
        message = str(body.get("msg") or body.get("message") or "")
        return body.get("error_code") == "email_exists" or (
            "already been registered" in message
        )

    def _auth_url(self, path: str) -> str:
        if not self._settings.supabase_url:
            raise ApiError(
                status_code=503,
                code="DATABASE_NOT_CONFIGURED",
                detail="Supabase n'est pas configuré côté API",
            )
        return f"{self._settings.supabase_url.rstrip('/')}/auth/v1/{path.lstrip('/')}"

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
        upstream_code = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                message = str(body.get("message") or "")
                upstream_code = str(body.get("code") or "")
        except ValueError:
            pass
        if response.status_code < 500 and message.isupper() and " " not in message:
            return ApiError(
                status_code=409,
                code=message,
                detail=message,
                context={"upstream_status": response.status_code},
            )
        context = {
            "upstream_status": response.status_code,
            "upstream_code": upstream_code,
            "upstream_message": message[:300],
        }
        # 57014 : statement_timeout de Postgres (8 s pour les rôles de l'API).
        if upstream_code == "57014":
            return ApiError(
                status_code=503,
                code="DATABASE_TIMEOUT",
                detail="Supabase a interrompu une requête trop longue",
                context=context,
            )
        return ApiError(
            status_code=503,
            code="DATABASE_UNAVAILABLE",
            detail="Supabase n'a pas pu traiter la requête",
            context=context,
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
