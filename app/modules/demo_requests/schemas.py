from pydantic import BaseModel, ConfigDict, Field, field_validator

# Même tolérance que le formulaire (scripts/demo-form.js) : qqch@qqch.tld.
EMAIL_PATTERN = r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$"


class DemoRequestPayload(BaseModel):
    """Formulaire de request-demo.html, tel que l'envoie demo-form.js."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    name: str = Field(min_length=1, max_length=200)
    organisation: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320, pattern=EMAIL_PATTERN)
    context: str | None = Field(default=None, max_length=5000)
    source: str | None = Field(default=None, max_length=200)
    # Honeypot : invisible pour un visiteur, rempli par les robots.
    website: str | None = Field(default=None, max_length=2000)

    @field_validator("email")
    @classmethod
    def lowercase_email(cls, value: str) -> str:
        return value.lower()

    @field_validator("context", "source", "website")
    @classmethod
    def empty_as_none(cls, value: str | None) -> str | None:
        return value or None


class DemoRequestReceipt(BaseModel):
    status: str = "received"
