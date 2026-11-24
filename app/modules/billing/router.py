from typing import Annotated
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.billing.invoice_pdf import Seller
from app.modules.billing.repository import BillingRepository
from app.modules.billing.schemas import (
    BillingContactRequest,
    CheckoutRequest,
    SessionUrl,
)
from app.modules.billing.service import (
    BillingService,
    StripeWebhookHandler,
    stripe_gateway,
)

router = APIRouter(prefix="/billing", tags=["billing"])


def get_billing_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> BillingService:
    client: httpx.AsyncClient = request.app.state.http_client
    return BillingService(
        BillingRepository(SupabaseGateway(settings, client)),
        stripe_gateway(client, settings),
        dashboard_url=settings.dashboard_url,
        seller=Seller(
            name=settings.billing_seller_name,
            address=settings.billing_seller_address,
            siren=settings.billing_seller_siren,
            vat=settings.billing_seller_vat,
            iban=settings.billing_seller_iban,
            email=settings.billing_seller_email,
        ),
    )


def get_webhook_handler(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> StripeWebhookHandler:
    client: httpx.AsyncClient = request.app.state.http_client
    return StripeWebhookHandler(
        BillingRepository(SupabaseGateway(settings, client)),
        stripe_gateway(client, settings),
        webhook_secret=settings.stripe_webhook_secret,
        tolerance_seconds=settings.stripe_webhook_tolerance_seconds,
    )


@router.post("/checkout-session", response_model=SessionUrl)
async def create_checkout_session(
    payload: CheckoutRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[BillingService, Depends(get_billing_service)],
) -> SessionUrl:
    """Souscription en ligne d'une offre (administrateur de l'organisation)."""
    return SessionUrl(url=await service.checkout(payload.plan_code, user))


@router.post("/portal-session", response_model=SessionUrl)
async def create_portal_session(
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[BillingService, Depends(get_billing_service)],
) -> SessionUrl:
    """Portail Stripe : moyen de paiement, factures, résiliation."""
    return SessionUrl(url=await service.portal(user))


@router.post("/contact")
async def update_billing_contact(
    payload: BillingContactRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[BillingService, Depends(get_billing_service)],
) -> dict[str, bool]:
    """Coordonnées de facturation ; l'e-mail suit chez Stripe."""
    await service.update_contact(payload, user)
    return {"saved": True}


@router.get(
    "/invoices/{invoice_id}/pdf",
    response_class=Response,
    responses={
        200: {"description": "PDF de la facture"},
        403: {"description": "Réservé aux administrateurs de l'organisation"},
        404: {"description": "Facture introuvable"},
    },
)
async def download_invoice(
    invoice_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[BillingService, Depends(get_billing_service)],
) -> Response:
    invoice = await service.invoice_pdf(invoice_id, user)
    return Response(
        content=invoice.content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{invoice.filename}"'},
    )


@router.post("/webhook", status_code=200, include_in_schema=False)
async def stripe_webhook(
    request: Request,
    handler: Annotated[StripeWebhookHandler, Depends(get_webhook_handler)],
) -> dict[str, bool]:
    """Point d'entrée des événements Stripe : signature vérifiée sur le corps
    brut, sans jeton utilisateur."""
    await handler.handle(await request.body(), request.headers.get("Stripe-Signature"))
    return {"received": True}
