"""Demandes de démo de la landing page.

Ordre volontaire : la demande est enregistrée AVANT l'e-mail. Si l'envoi
échoue (ou n'est pas configuré), elle reste visible dans le backoffice
(/demo-requests, badge « Non notifiée ») et le visiteur reçoit quand même
un succès : sa demande nous est bien parvenue.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Protocol
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.errors import ApiError
from app.modules.demo_requests.notifier import Mail
from app.modules.demo_requests.schemas import DemoRequestPayload

logger = logging.getLogger(__name__)

USER_AGENT_MAX = 500


class Repository(Protocol):
    async def count_recent(self, ip: str, *, since: datetime, limit: int) -> int: ...

    async def insert(self, row: dict) -> None: ...

    async def mark_notified(self, request_id: UUID) -> None: ...


class Notifier(Protocol):
    async def send(self, mail: Mail) -> None: ...


def _paris() -> tzinfo:
    try:
        return ZoneInfo("Europe/Paris")
    except ZoneInfoNotFoundError:
        return UTC


class DemoRequestService:
    def __init__(
        self,
        repository: Repository,
        notifier: Notifier | None,
        *,
        sender: str | None,
        recipients: list[str],
        backoffice_url: str | None = None,
        rate_limit: int = 3,
        rate_window_seconds: int = 600,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        # Envoi possible seulement si tout est renseigné.
        self._notifier = notifier if sender and recipients else None
        self._sender = sender or ""
        self._recipients = recipients
        self._backoffice_url = backoffice_url
        self._rate_limit = rate_limit
        self._rate_window = timedelta(seconds=rate_window_seconds)
        self._clock = clock

    async def submit(
        self,
        payload: DemoRequestPayload,
        *,
        ip: str | None,
        user_agent: str | None,
    ) -> bool:
        """True si la demande a été enregistrée (False : honeypot)."""
        # Honeypot rempli : on répond comme pour une vraie demande, pour que
        # le robot n'apprenne rien, sans rien stocker ni envoyer.
        if payload.website:
            logger.info("Demo request dropped by the honeypot")
            return False

        received_at = self._clock()
        if ip:
            await self._check_rate(ip, received_at)

        request_id = uuid4()
        await self._repository.insert(
            {
                "id": str(request_id),
                "name": payload.name,
                "organisation": payload.organisation,
                "email": payload.email,
                "context": payload.context,
                "source": payload.source,
                "user_agent": (user_agent or "")[:USER_AGENT_MAX] or None,
                "ip": ip,
            }
        )

        if self._notifier is None:
            logger.warning(
                "Demo request %s stored without notification: "
                "RESEND_API_KEY, DEMO_REQUEST_MAIL_FROM or DEMO_REQUEST_MAIL_TO "
                "is missing",
                request_id,
            )
            return True
        try:
            await self._notifier.send(self._mail(payload, received_at))
            await self._repository.mark_notified(request_id)
        except Exception:
            logger.exception(
                "Demo request %s stored but the notification failed", request_id
            )
        return True

    async def _check_rate(self, ip: str, now: datetime) -> None:
        try:
            recent = await self._repository.count_recent(
                ip, since=now - self._rate_window, limit=self._rate_limit
            )
        except ApiError:
            # Le contrôle de débit ne doit pas faire perdre une vraie demande.
            logger.exception("Demo request rate check failed")
            return
        if recent >= self._rate_limit:
            raise ApiError(
                status_code=429,
                code="DEMO_REQUEST_RATE_LIMITED",
                detail="Trop de demandes depuis cette adresse, réessayez plus tard",
                context={"retry_after_seconds": int(self._rate_window.total_seconds())},
            )

    def _mail(self, payload: DemoRequestPayload, received_at: datetime) -> Mail:
        # Les retours à la ligne n'ont rien à faire dans un objet d'e-mail.
        organisation = " ".join(payload.organisation.split())[:120]
        local = received_at.astimezone(_paris())
        lines = [
            "Nouvelle demande de démonstration depuis la landing page.",
            "",
            f"Nom : {payload.name}",
            f"Organisation : {payload.organisation}",
            f"E-mail : {payload.email}",
            "",
            "Ce qu'ils évaluent :",
            payload.context or "(non renseigné)",
            "",
            f"Reçue le {local:%d/%m/%Y à %H:%M} (heure de Paris)",
        ]
        if payload.source:
            lines.append(f"Page : {payload.source}")
        if self._backoffice_url:
            lines += ["", f"Suivi : {self._backoffice_url.rstrip('/')}/demo-requests"]
        lines += ["", "Répondre à ce message écrit directement au demandeur."]
        return Mail(
            sender=self._sender,
            recipients=self._recipients,
            reply_to=payload.email,
            subject=f"Demande de démo — {organisation}",
            text="\n".join(lines),
        )
