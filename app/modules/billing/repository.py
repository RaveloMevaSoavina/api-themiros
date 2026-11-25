"""Facturation côté base.

Les lectures de l'utilisateur passent par son jeton : les fonctions
organization_* vérifient qu'il est administrateur de son organisation. Les
écritures du webhook (stripe_*) passent par la clé serveur.
"""

from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway

# Erreurs métier des fonctions SQL → statut HTTP.
STATUS_BY_CODE = {
    "ORGANIZATION_ADMIN_REQUIRED": 403,
    "INVOICE_NOT_FOUND": 404,
    "BILLING_EMAIL_INVALID": 422,
}


def _remap(error: ApiError) -> ApiError:
    if error.status_code in {401, 403} and error.code == "DATABASE_ACCESS_DENIED":
        return ApiError(
            status_code=403,
            code="ORGANIZATION_ADMIN_REQUIRED",
            detail="Réservé aux administrateurs de l'organisation",
        )
    status = STATUS_BY_CODE.get(error.code)
    if status is not None:
        return ApiError(status_code=status, code=error.code, detail=error.detail)
    return error


class BillingRepository:
    def __init__(self, gateway: SupabaseGateway) -> None:
        self._gateway = gateway

    async def _user_rpc(self, name: str, *, token: str, payload: dict) -> Any:
        try:
            return await self._gateway.rpc_value(name, token=token, payload=payload)
        except ApiError as error:
            raise _remap(error) from error

    # -- Utilisateur ---------------------------------------------------------

    async def organization_billing(self, *, token: str) -> dict[str, Any]:
        return await self._user_rpc("organization_billing", token=token, payload={})

    async def update_contact(
        self,
        *,
        token: str,
        billing_email: str | None,
        billing_address: str | None,
        vat_number: str | None,
    ) -> None:
        await self._user_rpc(
            "organization_update_billing_contact",
            token=token,
            payload={
                "p_billing_email": billing_email,
                "p_billing_address": billing_address,
                "p_vat_number": vat_number,
            },
        )

    async def invoice(self, invoice_id: UUID, *, token: str) -> dict[str, Any]:
        return await self._user_rpc(
            "organization_billing_invoice",
            token=token,
            payload={"p_invoice_id": str(invoice_id)},
        )

    # -- Serveur -------------------------------------------------------------

    async def stripe_customer_id(self, organization_id: str) -> str | None:
        rows = await self._gateway.service_select(
            "organizations",
            params={"select": "stripe_customer_id", "id": f"eq.{organization_id}"},
        )
        return rows[0].get("stripe_customer_id") if rows else None

    async def plans(self) -> dict[str, dict[str, Any]]:
        rows = await self._gateway.service_select(
            "billing_plans",
            params={"select": "code,name,self_serve,stripe_lookup_key,active"},
        )
        return {row["code"]: row for row in rows}

    async def set_customer(self, organization_id: str, customer_id: str) -> str:
        return await self._gateway.service_rpc(
            "stripe_set_customer",
            payload={
                "p_organization_id": organization_id,
                "p_customer_id": customer_id,
            },
        )

    async def replace_customer(
        self, organization_id: str, stale_id: str, customer_id: str
    ) -> str | None:
        """Remplace un client Stripe disparu (supprimé, autre compte Stripe).
        Seulement s'il est encore en place : un Checkout concurrent qui l'a
        déjà remplacé garde le sien. Renvoie le client finalement en place."""
        await self._gateway.service_update(
            "organizations",
            params={
                "id": f"eq.{organization_id}",
                "stripe_customer_id": f"eq.{stale_id}",
            },
            payload={"stripe_customer_id": customer_id},
        )
        return await self.stripe_customer_id(organization_id)

    async def organization_for_customer(self, customer_id: str) -> str | None:
        return await self._gateway.service_rpc(
            "stripe_organization_for_customer",
            payload={"p_customer_id": customer_id},
        )

    async def begin_event(self, event_id: str, event_type: str) -> bool:
        return bool(
            await self._gateway.service_rpc(
                "stripe_begin_event", payload={"p_id": event_id, "p_type": event_type}
            )
        )

    async def finish_event(self, event_id: str, error: str | None = None) -> None:
        await self._gateway.service_rpc(
            "stripe_finish_event", payload={"p_id": event_id, "p_error": error}
        )

    async def sync_subscription(
        self,
        *,
        organization_id: str,
        stripe_subscription_id: str,
        stripe_price_id: str | None,
        plan_code: str,
        status: str,
        price_eur: Decimal,
        starts_on: date,
        ends_on: date,
        cancel_at_period_end: bool,
    ) -> None:
        await self._gateway.service_rpc(
            "stripe_sync_subscription",
            payload={
                "p_organization_id": organization_id,
                "p_stripe_subscription_id": stripe_subscription_id,
                "p_stripe_price_id": stripe_price_id,
                "p_plan_code": plan_code,
                "p_status": status,
                "p_price_eur": str(price_eur),
                "p_starts_on": starts_on.isoformat(),
                "p_ends_on": ends_on.isoformat(),
                "p_cancel_at_period_end": cancel_at_period_end,
            },
        )

    async def sync_invoice(
        self,
        *,
        organization_id: str,
        stripe_invoice_id: str,
        stripe_subscription_id: str | None,
        number: str,
        label: str,
        amount_eur: Decimal,
        tax_eur: Decimal,
        total_eur: Decimal,
        status: str,
        issued_on: date,
        due_on: date | None,
        paid_on: date | None,
    ) -> None:
        await self._gateway.service_rpc(
            "stripe_sync_invoice",
            payload={
                "p_organization_id": organization_id,
                "p_stripe_invoice_id": stripe_invoice_id,
                "p_stripe_subscription_id": stripe_subscription_id,
                "p_number": number,
                "p_label": label,
                "p_amount_eur": str(amount_eur),
                "p_tax_eur": str(tax_eur),
                "p_total_eur": str(total_eur),
                "p_status": status,
                "p_issued_on": issued_on.isoformat(),
                "p_due_on": due_on.isoformat() if due_on else None,
                "p_paid_on": paid_on.isoformat() if paid_on else None,
            },
        )

    async def sync_customer(
        self,
        *,
        customer_id: str,
        billing_email: str | None,
        billing_address: str | None,
        vat_number: str | None,
    ) -> None:
        await self._gateway.service_rpc(
            "stripe_sync_customer",
            payload={
                "p_customer_id": customer_id,
                "p_billing_email": billing_email,
                "p_billing_address": billing_address,
                "p_vat_number": vat_number,
            },
        )
