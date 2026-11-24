"""Crée dans Stripe les produits et prix des offres en libre-service.

    python -m app.modules.billing.bootstrap

Idempotent : un prix déjà présent (même lookup_key) est laissé tel quel. Les
montants reprennent le prix catalogue de `billing_plans` (HT, par an). Pour
changer un prix, créer un nouveau prix dans Stripe et y transférer le
lookup_key (« Transfer lookup key ») : les abonnements en cours gardent
l'ancien.

Le reste se règle dans le tableau de bord Stripe (voir DEPLOYMENT.md) :
Stripe Tax, préfixe des factures, portail client, webhook.
"""

import asyncio

import httpx

from app.core.config import get_settings
from app.modules.billing.stripe_gateway import StripeGateway

PLANS = (
    ("essential", "Themiros Essentiel", 408000),
    ("professional", "Themiros Professionnel", 1500000),
)


async def bootstrap() -> None:
    settings = get_settings()
    if not settings.stripe_secret_key:
        raise SystemExit("STRIPE_SECRET_KEY manquant")
    mode = "live" if settings.stripe_secret_key.startswith("sk_live") else "test"
    print(f"Stripe ({mode})")

    async with httpx.AsyncClient() as client:
        stripe = StripeGateway(
            client,
            secret_key=settings.stripe_secret_key,
            api_version=settings.stripe_api_version,
        )
        for code, name, unit_amount in PLANS:
            lookup_key = f"themiros_{code}_annual"
            existing = await stripe._request(
                "GET", "prices", params={"lookup_keys": [lookup_key]}
            )
            if existing.get("data"):
                price = existing["data"][0]
                print(f"  {lookup_key}: déjà présent ({price['id']})")
                continue
            product = await stripe._request(
                "POST",
                "products",
                params={
                    "name": name,
                    "metadata": {"plan_code": code},
                    "tax_code": "txcd_10103001",  # SaaS, usage professionnel
                },
                idempotency_key=f"product-{code}",
            )
            price = await stripe._request(
                "POST",
                "prices",
                params={
                    "product": product["id"],
                    "currency": "eur",
                    "unit_amount": unit_amount,
                    "recurring": {"interval": "year"},
                    "tax_behavior": "exclusive",
                    "lookup_key": lookup_key,
                    "nickname": f"{name} (annuel)",
                },
                idempotency_key=f"price-{lookup_key}",
            )
            print(f"  {lookup_key}: créé ({price['id']})")


if __name__ == "__main__":
    asyncio.run(bootstrap())
