"""Facturation : souscription Stripe, portail, PDF et miroir des webhooks.

Stripe fait foi pour les abonnements souscrits en ligne ; la base en garde le
miroir (subscriptions / invoices, source = 'stripe') que lit la page
Facturation. Le webhook relit toujours l'objet chez Stripe au lieu de se fier
au contenu de l'événement : les événements arrivent dans le désordre, l'état
relu est le dernier.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

from app.api.dependencies import CurrentUser
from app.core.errors import ApiError
from app.modules.billing.invoice_pdf import Seller, render_invoice_pdf
from app.modules.billing.repository import BillingRepository
from app.modules.billing.schemas import BillingContactRequest
from app.modules.billing.stripe_gateway import StripeGateway, verify_webhook
from app.modules.ingestion.domain import country_name

logger = logging.getLogger(__name__)

# Statut Stripe → statut interne. None : pas de miroir (abonnement dont le
# premier paiement n'a pas abouti).
SUBSCRIPTION_STATUS = {
    "trialing": "trialing",
    "active": "active",
    "past_due": "past_due",
    "unpaid": "past_due",
    "paused": "past_due",
    "canceled": "canceled",
    "incomplete_expired": "canceled",
    "incomplete": None,
}
# Les brouillons ne sont pas recopiés ; une facture irrécouvrable reste due.
INVOICE_STATUS = {
    "open": "issued",
    "paid": "paid",
    "void": "void",
    "uncollectible": "issued",
}


class StripeApi(Protocol):
    async def price(self, lookup_key: str) -> dict[str, Any]: ...
    async def create_customer(
        self, *, organization_id: str, name: str, email: str | None
    ) -> dict[str, Any]: ...
    async def update_customer(self, customer_id: str, **params: Any) -> None: ...
    async def create_checkout_session(
        self, params: dict[str, Any]
    ) -> dict[str, Any]: ...
    async def create_portal_session(
        self, *, customer_id: str, return_url: str
    ) -> dict[str, Any]: ...
    async def subscription(self, subscription_id: str) -> dict[str, Any]: ...
    async def invoice(self, invoice_id: str) -> dict[str, Any]: ...
    async def customer(self, customer_id: str) -> dict[str, Any]: ...
    async def checkout_session(self, session_id: str) -> dict[str, Any]: ...
    async def download(self, url: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class InvoiceFile:
    filename: str
    content: bytes


def _not_configured() -> ApiError:
    return ApiError(
        status_code=503,
        code="BILLING_NOT_CONFIGURED",
        detail="La souscription en ligne n'est pas configurée",
    )


def format_address(address: dict[str, Any] | None) -> str:
    """Adresse postale Stripe en lignes, à la française : rue, complément,
    « code postal ville », région, puis le pays en toutes lettres
    (« MG » → « Madagascar »). Lignes vides et doublons écartés."""
    address = address or {}

    def clean(key: str) -> str:
        return " ".join(str(address.get(key) or "").split())

    city = clean("city")
    locality = " ".join(part for part in (clean("postal_code"), city) if part)
    state = clean("state")
    code = clean("country").upper()
    country = (country_name(code) or code) if code else ""
    lines = [
        clean("line1"),
        clean("line2"),
        locality,
        state if state.casefold() != city.casefold() else "",
        country,
    ]
    unique: list[str] = []
    for line in lines:
        if line and line.casefold() not in {seen.casefold() for seen in unique}:
            unique.append(line)
    return "\n".join(unique)


async def _customer_exists(stripe: StripeApi, customer_id: str) -> bool:
    """Faux si Stripe ne connaît pas (ou plus) ce client ; toute autre erreur
    remonte telle quelle."""
    try:
        customer = await stripe.customer(customer_id)
    except ApiError as error:
        if error.context.get("upstream_status") == 404:
            return False
        raise
    return not customer.get("deleted")


def _day(timestamp: int | None) -> date | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).date()


def _euros(cents: int | None) -> Decimal:
    return (Decimal(cents or 0) / 100).quantize(Decimal("0.01"))


def _id(value: Any) -> str | None:
    """Identifiant d'un champ Stripe, développé (objet) ou non (chaîne)."""
    if isinstance(value, dict):
        return value.get("id")
    return value


class BillingService:
    def __init__(
        self,
        repository: BillingRepository,
        stripe: StripeApi | None,
        *,
        dashboard_url: str,
        seller: Seller,
    ) -> None:
        self._repository = repository
        self._stripe = stripe
        self._billing_url = f"{dashboard_url.rstrip('/')}/billing"
        self._seller = seller

    def _require_stripe(self) -> StripeApi:
        if self._stripe is None:
            raise _not_configured()
        return self._stripe

    async def checkout(self, plan_code: str, user: CurrentUser) -> str:
        """Session Checkout d'une offre en libre-service. Les URL de retour
        sont construites ici, jamais reçues du navigateur."""
        stripe = self._require_stripe()
        billing = await self._repository.organization_billing(token=user.token)
        organization = billing["organization"]
        organization_id = organization["id"]

        plan = (await self._repository.plans()).get(plan_code)
        if not plan or not plan.get("active") or not plan.get("self_serve"):
            raise ApiError(
                status_code=409,
                code="PLAN_NOT_SELF_SERVE",
                detail="Cette offre se souscrit sur devis",
            )
        if not plan.get("stripe_lookup_key"):
            raise _not_configured()

        subscription = billing.get("subscription")
        if subscription and subscription["status"] != "canceled":
            if subscription["source"] == "stripe":
                raise ApiError(
                    status_code=409,
                    code="ALREADY_SUBSCRIBED",
                    detail="Abonnement déjà en cours : gérez-le depuis le portail",
                )
            # Un pilote (trialing) sera clos à l'activation ; tout autre
            # contrat négocié se modifie avec l'équipe Themiros.
            if subscription["status"] != "trialing":
                raise ApiError(
                    status_code=409,
                    code="MANUAL_CONTRACT_ACTIVE",
                    detail="Votre contrat en cours se modifie avec l'équipe Themiros",
                )

        price = await stripe.price(plan["stripe_lookup_key"])
        customer_id = await self._customer(stripe, organization)
        metadata = {"organization_id": organization_id, "plan_code": plan_code}
        session = await stripe.create_checkout_session(
            {
                "mode": "subscription",
                "customer": customer_id,
                "client_reference_id": organization_id,
                "line_items": [{"price": price["id"], "quantity": 1}],
                "metadata": metadata,
                "subscription_data": {"metadata": metadata},
                "automatic_tax": {"enabled": True},
                "tax_id_collection": {"enabled": True},
                "billing_address_collection": "required",
                "customer_update": {"name": "auto", "address": "auto"},
                "locale": "auto",
                "success_url": f"{self._billing_url}?checkout=success",
                "cancel_url": f"{self._billing_url}?checkout=canceled",
            }
        )
        return session["url"]

    async def _customer(self, stripe: StripeApi, organization: dict[str, Any]) -> str:
        """Client Stripe de l'organisation, créé à son premier Checkout. Un
        client enregistré mais disparu chez Stripe (supprimé, ou clé d'un
        autre compte) est recréé et remplacé, sans intervention manuelle."""
        organization_id = organization["id"]
        existing = await self._repository.stripe_customer_id(organization_id)
        if existing and await _customer_exists(stripe, existing):
            return existing
        created = await stripe.create_customer(
            organization_id=organization_id,
            name=organization["name"],
            email=organization.get("billing_email"),
        )
        if existing:
            logger.warning(
                "Client Stripe %s introuvable pour l'organisation %s : remplacé par %s",
                existing,
                organization_id,
                created["id"],
            )
            replaced = await self._repository.replace_customer(
                organization_id, existing, created["id"]
            )
            return replaced or created["id"]
        # Deux admins en même temps : la base garde le premier client.
        return await self._repository.set_customer(organization_id, created["id"])

    async def portal(self, user: CurrentUser) -> str:
        stripe = self._require_stripe()
        billing = await self._repository.organization_billing(token=user.token)
        customer_id = await self._repository.stripe_customer_id(
            billing["organization"]["id"]
        )
        if not customer_id:
            raise ApiError(
                status_code=409,
                code="BILLING_CUSTOMER_MISSING",
                detail="Aucun paiement en ligne n'est encore associé à l'organisation",
            )
        session = await stripe.create_portal_session(
            customer_id=customer_id, return_url=self._billing_url
        )
        return session["url"]

    async def update_contact(
        self, payload: BillingContactRequest, user: CurrentUser
    ) -> None:
        await self._repository.update_contact(
            token=user.token,
            billing_email=payload.billing_email,
            billing_address=payload.billing_address,
            vat_number=payload.vat_number,
        )
        # L'e-mail des factures Stripe suit ; l'adresse et la TVA se
        # modifient dans le portail (elles reviennent par customer.updated).
        if self._stripe is None or not payload.billing_email:
            return
        billing = await self._repository.organization_billing(token=user.token)
        customer_id = await self._repository.stripe_customer_id(
            billing["organization"]["id"]
        )
        if customer_id:
            await self._stripe.update_customer(
                customer_id, email=billing["organization"]["billing_email"]
            )

    async def invoice_pdf(self, invoice_id: UUID, user: CurrentUser) -> InvoiceFile:
        invoice = await self._repository.invoice(invoice_id, token=user.token)
        filename = f"{invoice.get('number') or invoice_id}.pdf"
        if invoice.get("source") == "stripe":
            stripe = self._require_stripe()
            # Le lien du PDF est signé et peut expirer : on le relit.
            remote = await stripe.invoice(invoice["stripe_invoice_id"])
            if not remote.get("invoice_pdf"):
                raise ApiError(
                    status_code=409,
                    code="INVOICE_PDF_UNAVAILABLE",
                    detail="Le PDF de cette facture n'est pas encore disponible",
                )
            return InvoiceFile(filename, await stripe.download(remote["invoice_pdf"]))
        return InvoiceFile(filename, render_invoice_pdf(invoice, self._seller))


class StripeWebhookHandler:
    def __init__(
        self,
        repository: BillingRepository,
        stripe: StripeApi | None,
        *,
        webhook_secret: str | None,
        tolerance_seconds: int = 300,
    ) -> None:
        self._repository = repository
        self._stripe = stripe
        self._secret = webhook_secret
        self._tolerance = tolerance_seconds
        self._plans: dict[str, dict[str, Any]] | None = None

    async def handle(self, payload: bytes, signature: str | None) -> None:
        if self._stripe is None or not self._secret:
            raise _not_configured()
        event = verify_webhook(
            payload, signature, self._secret, tolerance=self._tolerance
        )
        event_id, event_type = event["id"], event["type"]
        if not await self._repository.begin_event(event_id, event_type):
            return
        try:
            await self._dispatch(event_type, event.get("data", {}).get("object", {}))
        except Exception as error:
            logger.exception("Webhook Stripe %s (%s) en échec", event_id, event_type)
            await self._repository.finish_event(event_id, str(error)[:1000])
            raise ApiError(
                status_code=500,
                code="STRIPE_WEBHOOK_FAILED",
                detail="Le webhook Stripe n'a pas pu être traité",
            ) from error
        await self._repository.finish_event(event_id)

    async def _dispatch(self, event_type: str, obj: dict[str, Any]) -> None:
        if event_type == "checkout.session.completed":
            if obj.get("mode") == "subscription" and obj.get("subscription"):
                await self.sync_subscription(
                    _id(obj["subscription"]), hint=obj.get("client_reference_id")
                )
            # Adresse et n° de TVA saisis dans Checkout : recopiés sans
            # attendre un customer.updated.
            if obj.get("customer"):
                await self.sync_customer(_id(obj["customer"]))
        elif event_type.startswith("customer.subscription."):
            await self.sync_subscription(obj["id"])
        elif event_type.startswith("invoice."):
            await self.sync_invoice(obj["id"])
        elif event_type == "customer.updated":
            await self.sync_customer(obj["id"])

    async def _organization(
        self, customer_id: str | None, metadata: dict[str, Any], hint: str | None = None
    ) -> str | None:
        organization_id = metadata.get("organization_id") or hint
        if not organization_id and customer_id:
            organization_id = await self._repository.organization_for_customer(
                customer_id
            )
        return organization_id

    async def _plan_code(self, lookup_key: str | None, metadata: dict[str, Any]) -> str:
        if self._plans is None:
            self._plans = await self._repository.plans()
        for code, plan in self._plans.items():
            if lookup_key and plan.get("stripe_lookup_key") == lookup_key:
                return code
        code = metadata.get("plan_code")
        if code in self._plans:
            return code
        raise ValueError(f"Offre introuvable pour le prix Stripe {lookup_key!r}")

    async def sync_subscription(
        self, subscription_id: str | None, *, hint: str | None = None
    ) -> None:
        if not subscription_id:
            return
        subscription = await self._stripe.subscription(subscription_id)
        status = SUBSCRIPTION_STATUS.get(subscription.get("status"))
        if status is None:
            return
        metadata = subscription.get("metadata") or {}
        organization_id = await self._organization(
            _id(subscription.get("customer")), metadata, hint
        )
        if not organization_id:
            logger.warning("Abonnement Stripe %s sans organisation", subscription_id)
            return

        items = (subscription.get("items") or {}).get("data") or []
        if not items:
            raise ValueError(f"Abonnement Stripe {subscription_id} sans ligne")
        item = items[0]
        price = item.get("price") or {}
        # Depuis l'API 2025-03-31.basil, la période est portée par la ligne.
        start = item.get("current_period_start") or subscription.get(
            "current_period_start"
        )
        end = item.get("current_period_end") or subscription.get("current_period_end")
        starts_on, ends_on = _day(start), _day(end)
        if starts_on is None or ends_on is None:
            raise ValueError(f"Abonnement Stripe {subscription_id} sans période")
        if ends_on <= starts_on:
            ends_on = date.fromordinal(starts_on.toordinal() + 1)

        await self._repository.sync_subscription(
            organization_id=organization_id,
            stripe_subscription_id=subscription_id,
            stripe_price_id=price.get("id"),
            plan_code=await self._plan_code(price.get("lookup_key"), metadata),
            status=status,
            price_eur=_euros(
                (price.get("unit_amount") or 0) * (item.get("quantity") or 1)
            ),
            starts_on=starts_on,
            ends_on=ends_on,
            cancel_at_period_end=bool(subscription.get("cancel_at_period_end")),
        )

    async def sync_invoice(self, invoice_id: str) -> None:
        invoice = await self._stripe.invoice(invoice_id)
        status = INVOICE_STATUS.get(invoice.get("status"))
        if status is None or not invoice.get("number"):
            return
        customer_id = _id(invoice.get("customer"))
        organization_id = (
            await self._repository.organization_for_customer(customer_id)
            if customer_id
            else None
        )
        if not organization_id:
            logger.warning("Facture Stripe %s sans organisation", invoice_id)
            return

        # Abonnement : `parent.subscription_details` depuis basil, champ
        # `subscription` avant.
        parent = (invoice.get("parent") or {}).get("subscription_details") or {}
        subscription_id = _id(parent.get("subscription")) or _id(
            invoice.get("subscription")
        )
        # L'abonnement est relu à chaque facture : un événement
        # customer.subscription.* perdu (ou non coché sur le point d'entrée)
        # ne laisse pas l'offre vide. Son échec ne bloque pas la facture.
        if subscription_id:
            try:
                await self.sync_subscription(subscription_id)
            except Exception:
                logger.exception(
                    "Abonnement Stripe %s non resynchronisé (facture %s)",
                    subscription_id,
                    invoice_id,
                )
        transitions = invoice.get("status_transitions") or {}
        total = _euros(invoice.get("total"))
        amount = _euros(
            invoice.get("total_excluding_tax")
            if invoice.get("total_excluding_tax") is not None
            else invoice.get("subtotal")
        )
        lines = (invoice.get("lines") or {}).get("data") or []
        label = (
            lines[0].get("description") if lines else None
        ) or "Abonnement Themiros"

        await self._repository.sync_invoice(
            organization_id=organization_id,
            stripe_invoice_id=invoice_id,
            stripe_subscription_id=subscription_id,
            number=invoice["number"],
            label=label,
            amount_eur=amount,
            tax_eur=total - amount,
            total_eur=total,
            status=status,
            issued_on=_day(transitions.get("finalized_at") or invoice.get("created")),
            due_on=_day(invoice.get("due_date")),
            paid_on=_day(transitions.get("paid_at")),
        )

    async def sync_customer(self, customer_id: str) -> None:
        customer = await self._stripe.customer(customer_id)
        formatted = format_address(customer.get("address"))
        tax_ids = (customer.get("tax_ids") or {}).get("data") or []
        vat = next(
            (tax["value"] for tax in tax_ids if tax.get("type") == "eu_vat"),
            tax_ids[0]["value"] if tax_ids else None,
        )
        await self._repository.sync_customer(
            customer_id=customer_id,
            billing_email=customer.get("email"),
            billing_address=formatted or None,
            vat_number=vat,
        )


def stripe_gateway(client, settings) -> StripeGateway | None:
    if not settings.stripe_secret_key:
        return None
    return StripeGateway(
        client,
        secret_key=settings.stripe_secret_key,
        api_version=settings.stripe_api_version,
    )
