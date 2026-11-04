"""Protection de la route publique des demandes de démo contre les robots.

Trois barrières, en mémoire du processus (uvicorn tourne en un seul
processus ; un redémarrage remet les compteurs à zéro, la limite en base du
service prend alors le relais) :

1. Tentatives par adresse : TOUTE requête compte, valide ou non, honeypot
   compris. Au-delà de `attempt_limit` sur `attempt_window`, l'adresse est
   bannie `ban_seconds`. Un robot qui martèle est donc coupé avant même que
   son corps soit lu, et ne coûte plus aucune requête à Supabase.
2. Honeypot : une adresse qui le remplit est un robot, bannie d'office. La
   réponse reste un succès pour ne rien lui apprendre ; c'est la suivante
   qui est refusée.
3. Plafond global : au plus `global_limit` demandes enregistrées par heure,
   toutes adresses confondues, contre un robot réparti sur beaucoup d'IP.

En IPv6, la clé est le /64 : un seul abonné en dispose entier et peut en
faire tourner les adresses à volonté.
"""

import ipaddress
import logging
import time
from collections import deque
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

PATH = "/api/v1/demo-requests"
# Le formulaire envoie quelques champs : bien en dessous de 16 Kio.
MAX_BODY_BYTES = 16 * 1024
GLOBAL_WINDOW_SECONDS = 3600


def client_ip(request: Request) -> str | None:
    """Adresse du visiteur. Derrière Caddy, c'est la dernière entrée de
    X-Forwarded-For (celle que Caddy ajoute ; les précédentes viennent du
    client et ne sont pas fiables). On ne garde qu'une adresse valide : la
    colonne est de type inet."""
    forwarded = request.headers.get("x-forwarded-for", "")
    candidates = [part.strip() for part in forwarded.split(",") if part.strip()]
    candidate = (
        candidates[-1]
        if candidates
        else (request.client.host if request.client else "")
    )
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def client_key(ip: str | None) -> str:
    if ip is None:
        return "unknown"
    address = ipaddress.ip_address(ip)
    if address.version == 6:
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


class AbuseGuard:
    def __init__(
        self,
        *,
        attempt_limit: int = 10,
        attempt_window_seconds: int = 600,
        ban_seconds: int = 3600,
        global_limit: int = 30,
        max_tracked: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._attempt_limit = attempt_limit
        self._attempt_window = attempt_window_seconds
        self._ban_seconds = ban_seconds
        self._global_limit = global_limit
        self._max_tracked = max_tracked
        self._clock = clock
        self._attempts: dict[str, deque[float]] = {}
        self._bans: dict[str, float] = {}
        self._stored: deque[float] = deque()

    def blocked_for(self, key: str) -> int | None:
        """Enregistre une tentative. Renvoie la durée du blocage restant en
        secondes, ou None si la requête peut passer."""
        now = self._clock()
        remaining = self._ban_remaining(key, now)
        if remaining is not None:
            return remaining

        attempts = self._attempts.setdefault(key, deque())
        attempts.append(now)
        while attempts and attempts[0] <= now - self._attempt_window:
            attempts.popleft()
        if len(attempts) > self._attempt_limit:
            self.ban(key, reason="too many attempts")
            return self._ban_seconds

        self._prune(now)
        return None

    def ban(self, key: str, *, reason: str) -> None:
        self._bans[key] = self._clock() + self._ban_seconds
        self._attempts.pop(key, None)
        logger.warning(
            "Demo requests: %s blocked for %ss (%s)", key, self._ban_seconds, reason
        )

    def global_retry_after(self) -> int | None:
        """Plafond global atteint : délai avant qu'une place se libère."""
        now = self._clock()
        self._expire_stored(now)
        if len(self._stored) < self._global_limit:
            return None
        return max(1, int(self._stored[0] + GLOBAL_WINDOW_SECONDS - now))

    def record_stored(self) -> None:
        self._stored.append(self._clock())

    def _expire_stored(self, now: float) -> None:
        while self._stored and self._stored[0] <= now - GLOBAL_WINDOW_SECONDS:
            self._stored.popleft()

    def _ban_remaining(self, key: str, now: float) -> int | None:
        until = self._bans.get(key)
        if until is None:
            return None
        if until <= now:
            del self._bans[key]
            return None
        return max(1, int(until - now))

    def _prune(self, now: float) -> None:
        """Borne la mémoire : un robot qui change d'adresse à chaque requête
        ne doit pas faire grossir les tables indéfiniment."""
        if len(self._attempts) > self._max_tracked:
            for key in [
                key
                for key, attempts in self._attempts.items()
                if not attempts or attempts[-1] <= now - self._attempt_window
            ]:
                del self._attempts[key]
        if len(self._attempts) > self._max_tracked:
            # Toujours trop : on oublie les plus anciennes (ordre d'insertion).
            for key in list(self._attempts)[: len(self._attempts) // 2]:
                del self._attempts[key]
        if len(self._bans) > self._max_tracked:
            for key in [key for key, until in self._bans.items() if until <= now]:
                del self._bans[key]


def too_many(code: str, retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content={
            "detail": "Trop de demandes, réessayez plus tard",
            "code": code,
            "context": {"retry_after_seconds": retry_after},
        },
    )


def install_demo_request_guard(app: FastAPI, guard: AbuseGuard) -> None:
    """À installer AVANT le CORSMiddleware (le dernier ajouté est le plus
    externe) : un 429 doit porter les en-têtes CORS, sinon le navigateur le
    masque et le formulaire affiche l'échec générique au lieu du message
    « patientez quelques minutes »."""
    app.state.demo_request_guard = guard

    @app.middleware("http")
    async def demo_request_guard(request: Request, call_next):
        if request.url.path.rstrip("/") != PATH or request.method != "POST":
            return await call_next(request)

        # Lu sur app.state à chaque requête : les tests le remplacent.
        current: AbuseGuard = request.app.state.demo_request_guard
        retry_after = current.blocked_for(client_key(client_ip(request)))
        if retry_after is not None:
            return too_many("DEMO_REQUEST_BLOCKED", retry_after)

        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Requête trop volumineuse",
                    "code": "PAYLOAD_TOO_LARGE",
                    "context": {"max_bytes": MAX_BODY_BYTES},
                },
            )
        return await call_next(request)
