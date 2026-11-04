"""Envoi de l'e-mail à l'équipe par l'API HTTP de Resend."""

from dataclasses import dataclass

import httpx

RESEND_URL = "https://api.resend.com/emails"


@dataclass(frozen=True, slots=True)
class Mail:
    sender: str
    recipients: list[str]
    reply_to: str
    subject: str
    text: str


class NotificationFailed(Exception):
    pass


class ResendNotifier:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    async def send(self, mail: Mail) -> None:
        try:
            response = await self._client.post(
                RESEND_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "from": mail.sender,
                    "to": mail.recipients,
                    "reply_to": mail.reply_to,
                    "subject": mail.subject,
                    "text": mail.text,
                },
                timeout=10,
            )
        except httpx.HTTPError as error:
            raise NotificationFailed(f"Resend unreachable: {error}") from error
        if not response.is_success:
            raise NotificationFailed(
                f"Resend {response.status_code}: {response.text[:300]}"
            )
