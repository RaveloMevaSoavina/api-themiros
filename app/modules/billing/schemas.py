from pydantic import BaseModel, Field


class CheckoutRequest(BaseModel):
    plan_code: str = Field(min_length=1, max_length=64)


class SessionUrl(BaseModel):
    """Page Stripe (Checkout ou portail) où rediriger le navigateur."""

    url: str


class BillingContactRequest(BaseModel):
    billing_email: str | None = Field(default=None, max_length=320)
    billing_address: str | None = Field(default=None, max_length=1000)
    vat_number: str | None = Field(default=None, max_length=32)
