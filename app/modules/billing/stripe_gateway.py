"""Accès à l'API Stripe, en httpx comme Supabase et Resend.

Pas de SDK : l'API Stripe est un REST à corps `application/x-www-form-urlencoded`
dont les paramètres imbriqués s'écrivent `a[b][0][c]=…` (voir `encode`). La
version d'API est figée par l'en-tête `Stripe-Version` : les objets relus par
le webhook ont toujours le même format.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from app.core.errors import ApiError

logger = logging.getLogger(__name__)

API_URL = "https://api.stripe.com/v1"


def encode(params: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Paramètres imbriqués au format Stripe : {"a": {"b": [1]}} → a[b][0]=1."""
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        name = f"{prefix}[{key}]" if prefix else str(key)
        if value is None:
            continue
        if isinstance(value, dict):
            pairs.extend(encode(value, name))
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    pairs.extend(encode(item, f"{name}[{index}]"))
                else:
                    pairs.append((f"{name}[{index}]", _scalar(item)))
        else:
            pairs.append((name, _scalar(value)))
    return pairs


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def verify_webhook(
    payload: bytes, signature_header: str | None, secret: str, *, tolerance: int
) -> dict[str, Any]:
    """Vérifie l'en-tête `Stripe-Signature` (t=…,v1=…) : HMAC-SHA256 de
    « t.corps » avec le secret du point d'entrée, horodatage dans la
    tolérance. Renvoie l'événement décodé."""
    invalid = ApiError(
        status_code=400,
        code="STRIPE_SIGNATURE_INVALID",
        detail="Signature du webhook Stripe invalide",
    )
    if not signature_header:
        raise invalid

    timestamp: int | None = None
    signatures: list[str] = []
    for part in signature_header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            try:
                timestamp = int(value)
            except ValueError:
                raise invalid from None
        elif key == "v1":
            signatures.append(value)
    if timestamp is None or not signatures:
        raise invalid
    if abs(time.time() - timestamp) > tolerance:
        raise invalid

    expected = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + payload, hashlib.sha256
    ).hexdigest()
    if not any(hmac.compare_digest(expected, signature) for signature in signatures):
        raise invalid

    try:
        event = json.loads(payload)
    except ValueError:
        raise invalid from None
    if not isinstance(event, dict) or "id" not in event or "type" not in event:
        raise invalid
    return event


class StripeGateway:
    def __init__(
        self, client: httpx.AsyncClient, *, secret_key: str, api_version: str
    ) -> None:
        self._client = client
        self._secret_key = secret_key
        self._api_version = api_version
        self._prices: dict[str, dict[str, Any]] = {}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._secret_key}",
            "Stripe-Version": self._api_version,
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        encoded = encode(params or {})
        try:
            response = await self._client.request(
                method,
                f"{API_URL}/{path.lstrip('/')}",
                params=encoded if method == "GET" else None,
                data=dict(encoded) if method != "GET" else None,
                headers=headers,
                timeout=20,
            )
        except httpx.HTTPError as error:
            raise ApiError(
                status_code=503,
                code="STRIPE_UNAVAILABLE",
                detail="Stripe est injoignable",
            ) from error

        if not response.is_success:
            body: dict[str, Any] = {}
            try:
                body = response.json().get("error") or {}
            except ValueError:
                pass
            logger.warning(
                "Stripe %s %s: %s %s %s",
                method,
                path,
                response.status_code,
                body.get("code") or body.get("type") or "",
                str(body.get("message") or "")[:300],
            )
            raise ApiError(
                status_code=502,
                code="STRIPE_REQUEST_FAILED",
                detail="Stripe a refusé la requête",
                context={
                    "upstream_status": response.status_code,
                    "upstream_code": str(body.get("code") or ""),
                },
            )
        return response.json()

    async def price(self, lookup_key: str) -> dict[str, Any]:
        """Prix actif portant ce lookup_key (mis en cache : il ne change pas
        sans que l'on change aussi le lookup_key)."""
        cached = self._prices.get(lookup_key)
        if cached is not None:
            return cached
        result = await self._request(
            "GET", "prices", params={"lookup_keys": [lookup_key], "active": True}
        )
        prices = result.get("data") or []
        if not prices:
            raise ApiError(
                status_code=503,
                code="STRIPE_PRICE_MISSING",
                detail="Le prix Stripe de cette offre n'est pas configuré",
                context={"lookup_key": lookup_key},
            )
        self._prices[lookup_key] = prices[0]
        return prices[0]

    async def create_customer(
        self, *, organization_id: str, name: str, email: str | None
    ) -> dict[str, Any]:
        # Pas de clé d'idempotence fixe par organisation : Stripe rejouerait
        # pendant 24 h la réponse du premier appel, même pour un client
        # supprimé depuis. Les doublons sont arbitrés par la base
        # (stripe_set_customer garde le premier client enregistré).
        return await self._request(
            "POST",
            "customers",
            params={
                "name": name,
                "email": email,
                "metadata": {"organization_id": organization_id},
            },
        )

    async def update_customer(self, customer_id: str, **params: Any) -> None:
        await self._request("POST", f"customers/{customer_id}", params=params)

    async def create_checkout_session(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "checkout/sessions", params=params)

    async def create_portal_session(
        self, *, customer_id: str, return_url: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "billing_portal/sessions",
            params={"customer": customer_id, "return_url": return_url},
        )

    async def subscription(self, subscription_id: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"subscriptions/{subscription_id}",
            params={"expand": ["items.data.price"]},
        )

    async def invoice(self, invoice_id: str) -> dict[str, Any]:
        return await self._request("GET", f"invoices/{invoice_id}")

    async def customer(self, customer_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"customers/{customer_id}", params={"expand": ["tax_ids"]}
        )

    async def checkout_session(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"checkout/sessions/{session_id}")

    async def download(self, url: str) -> bytes:
        """PDF d'une facture (`invoice_pdf`, lien signé de Stripe)."""
        try:
            response = await self._client.get(url, follow_redirects=True, timeout=30)
        except httpx.HTTPError as error:
            raise ApiError(
                status_code=503,
                code="STRIPE_UNAVAILABLE",
                detail="Stripe est injoignable",
            ) from error
        if not response.is_success:
            raise ApiError(
                status_code=502,
                code="STRIPE_INVOICE_PDF_FAILED",
                detail="Le PDF de la facture n'a pas pu être téléchargé chez Stripe",
                context={"upstream_status": response.status_code},
            )
        return response.content
