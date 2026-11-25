import hashlib
import hmac
import json
import time
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import CurrentUser, get_current_user
from app.core.errors import ApiError
from app.main import app
from app.modules.billing.invoice_pdf import Seller
from app.modules.billing.router import get_billing_service, get_webhook_handler
from app.modules.billing.service import BillingService, StripeWebhookHandler
from app.modules.billing.stripe_gateway import encode, verify_webhook

ADMIN = CurrentUser(
    id=UUID("22222222-2222-4222-8222-222222222222"),
    email="admin@client.org",
    token="admin-token",
)
ORG = "a0000000-0000-4000-8000-000000000001"
SECRET = "whsec_test"
SELLER = Seller(name="Evoranq", address="1 rue X\n75001 Paris", iban="FR76 0000")
PLANS = {
    "essential": {
        "code": "essential",
        "name": "Essentiel",
        "self_serve": True,
        "active": True,
        "stripe_lookup_key": "themiros_essential_annual",
    },
    "professional": {
        "code": "professional",
        "name": "Professionnel",
        "self_serve": True,
        "active": True,
        "stripe_lookup_key": "themiros_professional_annual",
    },
    "bespoke": {
        "code": "bespoke",
        "name": "Sur mesure",
        "self_serve": False,
        "active": True,
        "stripe_lookup_key": None,
    },
}


def ts(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, 12, tzinfo=UTC).timestamp())


class FakeRepository:
    def __init__(
        self,
        *,
        subscription: dict | None = None,
        customer: str | None = None,
        admin: bool = True,
        invoice: dict | None = None,
    ) -> None:
        self.subscription = subscription
        self.customer = customer
        self.admin = admin
        self._invoice = invoice
        self.events: dict[str, str | None] = {}
        self.finished: list[tuple[str, str | None]] = []
        self.subscriptions: list[dict] = []
        self.invoices: list[dict] = []
        self.customers: list[dict] = []
        self.contacts: list[dict] = []

    async def organization_billing(self, *, token):
        if not self.admin:
            raise ApiError(
                status_code=403, code="ORGANIZATION_ADMIN_REQUIRED", detail="x"
            )
        return {
            "organization": {
                "id": ORG,
                "name": "Organisation cliente",
                "billing_email": "compta@client.org",
                "has_stripe_customer": self.customer is not None,
            },
            "subscription": self.subscription,
        }

    async def update_contact(self, **kwargs):
        self.contacts.append(kwargs)

    async def invoice(self, invoice_id, *, token):
        if self._invoice is None:
            raise ApiError(status_code=404, code="INVOICE_NOT_FOUND", detail="x")
        return self._invoice

    async def stripe_customer_id(self, organization_id):
        return self.customer

    async def plans(self):
        return PLANS

    async def set_customer(self, organization_id, customer_id):
        self.customer = self.customer or customer_id
        return self.customer

    async def replace_customer(self, organization_id, stale_id, customer_id):
        if self.customer == stale_id:
            self.customer = customer_id
        return self.customer

    async def organization_for_customer(self, customer_id):
        return ORG if customer_id == "cus_1" else None

    async def begin_event(self, event_id, event_type):
        if event_id in self.events and self.events[event_id] == "done":
            return False
        self.events[event_id] = None
        return True

    async def finish_event(self, event_id, error=None):
        self.finished.append((event_id, error))
        if error is None:
            self.events[event_id] = "done"

    async def sync_subscription(self, **kwargs):
        self.subscriptions.append(kwargs)

    async def sync_invoice(self, **kwargs):
        self.invoices.append(kwargs)

    async def sync_customer(self, **kwargs):
        self.customers.append(kwargs)


class FakeStripe:
    def __init__(self) -> None:
        self.customers_created: list[dict] = []
        self.sessions: list[dict] = []
        self.portals: list[dict] = []
        self.updated: list[tuple[str, dict]] = []
        self.remote_subscriptions: dict[str, dict] = {}
        self.remote_invoices: dict[str, dict] = {}
        self.missing_customers: set[str] = set()
        self.deleted_customers: set[str] = set()
        self.fail = False

    async def price(self, lookup_key):
        return {"id": f"price_{lookup_key}", "lookup_key": lookup_key}

    async def create_customer(self, *, organization_id, name, email):
        self.customers_created.append({"organization_id": organization_id})
        return {"id": "cus_new"}

    async def update_customer(self, customer_id, **params):
        self.updated.append((customer_id, params))

    async def create_checkout_session(self, params):
        self.sessions.append(params)
        return {"url": "https://checkout.stripe.com/c/pay/cs_1"}

    async def create_portal_session(self, *, customer_id, return_url):
        self.portals.append({"customer": customer_id, "return_url": return_url})
        return {"url": "https://billing.stripe.com/p/session/1"}

    async def subscription(self, subscription_id):
        if self.fail:
            raise RuntimeError("Stripe en panne")
        return self.remote_subscriptions[subscription_id]

    async def invoice(self, invoice_id):
        return self.remote_invoices[invoice_id]

    async def customer(self, customer_id):
        if customer_id in self.missing_customers:
            raise ApiError(
                status_code=502,
                code="STRIPE_REQUEST_FAILED",
                detail="x",
                context={"upstream_status": 404, "upstream_code": "resource_missing"},
            )
        if customer_id in self.deleted_customers:
            return {"id": customer_id, "deleted": True}
        return {
            "id": customer_id,
            "email": "Compta@Client.org",
            "address": {
                "line1": "1 rue de la Paix",
                "postal_code": "75002",
                "city": "Paris",
                "country": "FR",
            },
            "tax_ids": {"data": [{"type": "eu_vat", "value": "FR12345678901"}]},
        }

    async def checkout_session(self, session_id):
        raise NotImplementedError

    async def download(self, url):
        return b"%PDF-stripe"


def service(repository, stripe=None) -> BillingService:
    return BillingService(
        repository, stripe, dashboard_url="https://app.themiros.com/", seller=SELLER
    )


def stripe_subscription(status="active", **overrides) -> dict:
    today = date(2026, 11, 23)
    return {
        "id": "sub_1",
        "status": status,
        "customer": "cus_1",
        "metadata": {"organization_id": ORG, "plan_code": "essential"},
        "cancel_at_period_end": False,
        "items": {
            "data": [
                {
                    "quantity": 1,
                    "current_period_start": ts(today),
                    "current_period_end": ts(date(2027, 11, 23)),
                    "price": {
                        "id": "price_ess",
                        "lookup_key": "themiros_essential_annual",
                        "unit_amount": 408000,
                    },
                }
            ]
        },
        **overrides,
    }


def signed(event: dict, *, secret: str = SECRET, at: int | None = None):
    body = json.dumps(event).encode()
    timestamp = at or int(time.time())
    signature = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()
    return body, f"t={timestamp},v1={signature}"


# ---------------------------------------------------------------------------
# Encodage et signature
# ---------------------------------------------------------------------------


def test_nested_parameters_use_the_stripe_bracket_syntax() -> None:
    assert encode(
        {
            "line_items": [{"price": "p", "quantity": 1}],
            "metadata": {"a": "b"},
            "expand": ["x"],
            "automatic_tax": {"enabled": True},
            "skip": None,
        }
    ) == [
        ("line_items[0][price]", "p"),
        ("line_items[0][quantity]", "1"),
        ("metadata[a]", "b"),
        ("expand[0]", "x"),
        ("automatic_tax[enabled]", "true"),
    ]


def test_a_valid_signature_returns_the_event() -> None:
    body, header = signed({"id": "evt_1", "type": "invoice.paid"})
    assert verify_webhook(body, header, SECRET, tolerance=300)["id"] == "evt_1"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "t=1,v1=abc",
        "v1=abc",
        "garbage",
    ],
)
def test_a_missing_or_forged_signature_is_refused(header) -> None:
    body, _ = signed({"id": "evt_1", "type": "x"})
    with pytest.raises(ApiError) as error:
        verify_webhook(body, header, SECRET, tolerance=300)
    assert error.value.code == "STRIPE_SIGNATURE_INVALID"


def test_a_replayed_old_signature_is_refused() -> None:
    body, header = signed({"id": "evt_1", "type": "x"}, at=int(time.time()) - 3600)
    with pytest.raises(ApiError):
        verify_webhook(body, header, SECRET, tolerance=300)


def test_a_signature_from_another_secret_is_refused() -> None:
    body, header = signed({"id": "evt_1", "type": "x"}, secret="whsec_other")
    with pytest.raises(ApiError):
        verify_webhook(body, header, SECRET, tolerance=300)


# ---------------------------------------------------------------------------
# Checkout et portail
# ---------------------------------------------------------------------------


async def test_checkout_creates_the_customer_once_and_builds_return_urls() -> None:
    repository, stripe = FakeRepository(), FakeStripe()

    url = await service(repository, stripe).checkout("essential", ADMIN)
    await service(repository, stripe).checkout("essential", ADMIN)

    assert url.startswith("https://checkout.stripe.com/")
    assert stripe.customers_created == [{"organization_id": ORG}]
    session = stripe.sessions[0]
    assert session["customer"] == "cus_new"
    assert session["line_items"] == [
        {"price": "price_themiros_essential_annual", "quantity": 1}
    ]
    assert session["subscription_data"]["metadata"] == {
        "organization_id": ORG,
        "plan_code": "essential",
    }
    assert session["success_url"] == "https://app.themiros.com/billing?checkout=success"
    assert session["cancel_url"] == "https://app.themiros.com/billing?checkout=canceled"
    assert session["automatic_tax"] == {"enabled": True}


@pytest.mark.parametrize("gone", ["missing_customers", "deleted_customers"])
async def test_checkout_replaces_a_customer_gone_from_stripe(gone) -> None:
    repository, stripe = FakeRepository(customer="cus_gone"), FakeStripe()
    getattr(stripe, gone).add("cus_gone")

    await service(repository, stripe).checkout("essential", ADMIN)

    assert stripe.customers_created == [{"organization_id": ORG}]
    assert repository.customer == "cus_new"
    assert stripe.sessions[0]["customer"] == "cus_new"


async def test_checkout_keeps_the_customer_on_other_stripe_errors() -> None:
    repository, stripe = FakeRepository(customer="cus_1"), FakeStripe()

    async def unavailable(customer_id):
        raise ApiError(status_code=503, code="STRIPE_UNAVAILABLE", detail="x")

    stripe.customer = unavailable
    with pytest.raises(ApiError) as error:
        await service(repository, stripe).checkout("essential", ADMIN)
    assert error.value.code == "STRIPE_UNAVAILABLE"
    assert stripe.customers_created == []
    assert repository.customer == "cus_1"


async def test_a_pilot_can_switch_to_self_serve() -> None:
    repository = FakeRepository(
        subscription={"status": "trialing", "source": "manual"}, customer="cus_1"
    )
    stripe = FakeStripe()
    await service(repository, stripe).checkout("professional", ADMIN)
    assert stripe.sessions[0]["customer"] == "cus_1"


@pytest.mark.parametrize(
    ("subscription", "plan", "code"),
    [
        ({"status": "active", "source": "stripe"}, "essential", "ALREADY_SUBSCRIBED"),
        ({"status": "past_due", "source": "stripe"}, "essential", "ALREADY_SUBSCRIBED"),
        (
            {"status": "active", "source": "manual"},
            "essential",
            "MANUAL_CONTRACT_ACTIVE",
        ),
        (None, "bespoke", "PLAN_NOT_SELF_SERVE"),
        (None, "unknown", "PLAN_NOT_SELF_SERVE"),
    ],
)
async def test_checkout_is_refused(subscription, plan, code) -> None:
    repository, stripe = FakeRepository(subscription=subscription), FakeStripe()
    with pytest.raises(ApiError) as error:
        await service(repository, stripe).checkout(plan, ADMIN)
    assert error.value.code == code
    assert error.value.status_code == 409
    assert stripe.sessions == []


async def test_a_non_admin_cannot_subscribe() -> None:
    with pytest.raises(ApiError) as error:
        await service(FakeRepository(admin=False), FakeStripe()).checkout(
            "essential", ADMIN
        )
    assert error.value.status_code == 403


async def test_without_stripe_keys_checkout_is_not_configured() -> None:
    with pytest.raises(ApiError) as error:
        await service(FakeRepository(), None).checkout("essential", ADMIN)
    assert error.value.code == "BILLING_NOT_CONFIGURED"
    assert error.value.status_code == 503


async def test_the_portal_requires_a_stripe_customer() -> None:
    with pytest.raises(ApiError) as error:
        await service(FakeRepository(), FakeStripe()).portal(ADMIN)
    assert error.value.code == "BILLING_CUSTOMER_MISSING"

    stripe = FakeStripe()
    url = await service(FakeRepository(customer="cus_1"), stripe).portal(ADMIN)
    assert url.startswith("https://billing.stripe.com/")
    assert stripe.portals == [
        {"customer": "cus_1", "return_url": "https://app.themiros.com/billing"}
    ]


# ---------------------------------------------------------------------------
# Factures
# ---------------------------------------------------------------------------

MANUAL_INVOICE = {
    "id": "i1",
    "number": "TH-2026-0001",
    "label": "Atelier de lancement",
    "amount_eur": 1000,
    "vat_rate": 20,
    "tax_eur": 200,
    "total_eur": 1200,
    "status": "paid",
    "issued_on": "2026-11-03",
    "due_on": "2026-12-03",
    "paid_on": "2026-11-10",
    "source": "manual",
    "stripe_invoice_id": None,
    "organization": {
        "name": "Organisation cliente",
        "billing_address": "2 avenue Y\n69001 Lyon",
        "vat_number": "FR98765432109",
        "billing_email": "compta@client.org",
    },
}


async def test_a_manual_invoice_is_rendered_as_pdf() -> None:
    file = await service(FakeRepository(invoice=MANUAL_INVOICE)).invoice_pdf(
        UUID(int=1), ADMIN
    )

    assert file.filename == "TH-2026-0001.pdf"
    assert file.content.startswith(b"%PDF")
    text = pymupdf.open("pdf", file.content)[0].get_text()
    assert "TH-2026-0001" in text
    assert "1\u202f200,00 €" in text
    assert "Acquittée le 10/11/2026" in text
    assert "FR76 0000" in text


async def test_a_stripe_invoice_pdf_is_relayed_from_stripe() -> None:
    stripe = FakeStripe()
    stripe.remote_invoices["in_1"] = {"invoice_pdf": "https://pay.stripe.com/pdf"}
    invoice = {
        **MANUAL_INVOICE,
        "number": "THS-0001",
        "source": "stripe",
        "stripe_invoice_id": "in_1",
    }

    file = await service(FakeRepository(invoice=invoice), stripe).invoice_pdf(
        UUID(int=1), ADMIN
    )

    assert file == type(file)("THS-0001.pdf", b"%PDF-stripe")


async def test_an_invoice_of_another_organization_is_not_found() -> None:
    with pytest.raises(ApiError) as error:
        await service(FakeRepository(invoice=None)).invoice_pdf(UUID(int=1), ADMIN)
    assert error.value.status_code == 404


async def test_contact_update_follows_to_the_stripe_customer() -> None:
    repository, stripe = FakeRepository(customer="cus_1"), FakeStripe()
    from app.modules.billing.schemas import BillingContactRequest

    await service(repository, stripe).update_contact(
        BillingContactRequest(billing_email="Compta@Client.org"), ADMIN
    )

    assert repository.contacts[0]["billing_email"] == "Compta@Client.org"
    assert stripe.updated == [("cus_1", {"email": "compta@client.org"})]


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def handler(repository, stripe) -> StripeWebhookHandler:
    return StripeWebhookHandler(repository, stripe, webhook_secret=SECRET)


async def test_a_subscription_is_mirrored_from_the_period_of_its_item() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription()
    body, header = signed(
        {
            "id": "evt_1",
            "type": "customer.subscription.updated",
            "data": {"object": {"id": "sub_1", "status": "incomplete"}},
        }
    )

    await handler(repository, stripe).handle(body, header)

    assert repository.subscriptions == [
        {
            "organization_id": ORG,
            "stripe_subscription_id": "sub_1",
            "stripe_price_id": "price_ess",
            "plan_code": "essential",
            "status": "active",
            "price_eur": Decimal("4080.00"),
            "starts_on": date(2026, 11, 23),
            "ends_on": date(2027, 11, 23),
            "cancel_at_period_end": False,
        }
    ]
    assert repository.finished == [("evt_1", None)]


async def test_a_duplicate_event_is_processed_once() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription()
    body, header = signed(
        {
            "id": "evt_1",
            "type": "customer.subscription.created",
            "data": {"object": {"id": "sub_1"}},
        }
    )

    await handler(repository, stripe).handle(body, header)
    await handler(repository, stripe).handle(body, header)

    assert len(repository.subscriptions) == 1


@pytest.mark.parametrize(
    ("stripe_status", "expected"),
    [
        ("trialing", "trialing"),
        ("active", "active"),
        ("past_due", "past_due"),
        ("unpaid", "past_due"),
        ("canceled", "canceled"),
        ("incomplete_expired", "canceled"),
        ("incomplete", None),
    ],
)
async def test_subscription_statuses_are_mapped(stripe_status, expected) -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription(stripe_status)

    await handler(repository, stripe).sync_subscription("sub_1")

    statuses = [row["status"] for row in repository.subscriptions]
    assert statuses == ([expected] if expected else [])


async def test_a_subscription_without_metadata_is_found_by_customer() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription(metadata={})
    await handler(repository, stripe).sync_subscription("sub_1")
    assert repository.subscriptions[0]["organization_id"] == ORG
    assert repository.subscriptions[0]["plan_code"] == "essential"


async def test_an_unknown_organization_is_logged_and_acknowledged() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription(
        metadata={}, customer="cus_unknown"
    )
    body, header = signed(
        {
            "id": "evt_1",
            "type": "customer.subscription.created",
            "data": {"object": {"id": "sub_1"}},
        }
    )
    await handler(repository, stripe).handle(body, header)
    assert repository.subscriptions == []
    assert repository.finished == [("evt_1", None)]


async def test_a_failure_is_recorded_and_returns_500_for_a_retry() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.fail = True
    body, header = signed(
        {
            "id": "evt_1",
            "type": "customer.subscription.created",
            "data": {"object": {"id": "sub_1"}},
        }
    )
    with pytest.raises(ApiError) as error:
        await handler(repository, stripe).handle(body, header)
    assert error.value.status_code == 500
    assert repository.finished == [("evt_1", "Stripe en panne")]


async def test_a_paid_invoice_is_mirrored_with_stripe_amounts() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription()
    stripe.remote_invoices["in_1"] = {
        "id": "in_1",
        "number": "THS-0001",
        "status": "paid",
        "customer": "cus_1",
        "parent": {"subscription_details": {"subscription": "sub_1"}},
        "total": 489600,
        "total_excluding_tax": 408000,
        "created": ts(date(2026, 11, 23)),
        "due_date": None,
        "status_transitions": {
            "finalized_at": ts(date(2026, 11, 23)),
            "paid_at": ts(date(2026, 11, 24)),
        },
        "lines": {"data": [{"description": "1 × Themiros Essentiel (annuel)"}]},
    }
    body, header = signed(
        {"id": "evt_2", "type": "invoice.paid", "data": {"object": {"id": "in_1"}}}
    )

    await handler(repository, stripe).handle(body, header)

    # L'abonnement de la facture est resynchronisé au passage.
    assert [row["stripe_subscription_id"] for row in repository.subscriptions] == [
        "sub_1"
    ]
    assert repository.invoices == [
        {
            "organization_id": ORG,
            "stripe_invoice_id": "in_1",
            "stripe_subscription_id": "sub_1",
            "number": "THS-0001",
            "label": "1 × Themiros Essentiel (annuel)",
            "amount_eur": Decimal("4080.00"),
            "tax_eur": Decimal("816.00"),
            "total_eur": Decimal("4896.00"),
            "status": "paid",
            "issued_on": date(2026, 11, 23),
            "due_on": None,
            "paid_on": date(2026, 11, 24),
        }
    ]


async def test_an_invoice_is_mirrored_even_if_its_subscription_fails() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.fail = True
    stripe.remote_invoices["in_1"] = {
        "id": "in_1",
        "number": "THS-0002",
        "status": "paid",
        "customer": "cus_1",
        "parent": {"subscription_details": {"subscription": "sub_1"}},
        "total": 408000,
        "total_excluding_tax": 408000,
        "created": ts(date(2026, 11, 23)),
        "status_transitions": {},
        "lines": {"data": []},
    }

    await handler(repository, stripe).sync_invoice("in_1")

    assert repository.subscriptions == []
    assert [row["number"] for row in repository.invoices] == ["THS-0002"]


async def test_a_draft_invoice_is_not_mirrored() -> None:
    repository, stripe = FakeRepository(), FakeStripe()
    stripe.remote_invoices["in_1"] = {"id": "in_1", "status": "draft", "number": None}
    await handler(repository, stripe).sync_invoice("in_1")
    assert repository.invoices == []


async def test_customer_details_come_back_from_the_portal() -> None:
    repository = FakeRepository()
    await handler(repository, FakeStripe()).sync_customer("cus_1")
    assert repository.customers == [
        {
            "customer_id": "cus_1",
            "billing_email": "Compta@Client.org",
            "billing_address": "1 rue de la Paix\n75002 Paris\nFR",
            "vat_number": "FR12345678901",
        }
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    repository, stripe = FakeRepository(customer="cus_1"), FakeStripe()
    stripe.remote_subscriptions["sub_1"] = stripe_subscription()
    app.dependency_overrides[get_current_user] = lambda: ADMIN
    app.dependency_overrides[get_billing_service] = lambda: service(repository, stripe)
    app.dependency_overrides[get_webhook_handler] = lambda: handler(repository, stripe)
    with TestClient(app) as test_client:
        yield test_client, repository
    app.dependency_overrides.clear()


def test_webhook_route_refuses_a_bad_signature(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/billing/webhook",
        content=b'{"id":"evt_1","type":"x"}',
        headers={"Stripe-Signature": "t=1,v1=bad"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "STRIPE_SIGNATURE_INVALID"


def test_webhook_route_accepts_a_signed_event(client) -> None:
    test_client, repository = client
    body, header = signed(
        {
            "id": "evt_9",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "mode": "subscription",
                    "subscription": "sub_1",
                    "client_reference_id": ORG,
                }
            },
        }
    )
    response = test_client.post(
        "/api/v1/billing/webhook", content=body, headers={"Stripe-Signature": header}
    )
    assert response.status_code == 200
    assert repository.subscriptions[0]["stripe_subscription_id"] == "sub_1"


def test_checkout_route_returns_the_stripe_url(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/billing/checkout-session", json={"plan_code": "essential"}
    )
    assert response.status_code == 200
    assert response.json() == {"url": "https://checkout.stripe.com/c/pay/cs_1"}


def test_contact_route_confirms_the_save(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/v1/billing/contact", json={"billing_email": "a@b.org"}
    )
    assert response.status_code == 200
    assert response.json() == {"saved": True}
