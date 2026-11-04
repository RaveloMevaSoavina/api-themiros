"""Accès à `public.demo_requests`, avec la clé serveur.

La table n'a aucune policy RLS : ni le navigateur ni un compte connecté n'y
accèdent. Le backoffice la lit par les fonctions backoffice_*_demo_request*.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from app.infrastructure.supabase import SupabaseGateway


class DemoRequestRepository:
    def __init__(self, gateway: SupabaseGateway) -> None:
        self._gateway = gateway

    async def count_recent(self, ip: str, *, since: datetime, limit: int) -> int:
        """Demandes de cette adresse depuis `since`, comptées jusqu'à `limit`."""
        rows = await self._gateway.service_select(
            "demo_requests",
            params={
                "select": "id",
                "ip": f"eq.{ip}",
                "created_at": f"gte.{since.isoformat()}",
                "limit": str(limit),
            },
        )
        return len(rows)

    async def insert(self, row: dict[str, Any]) -> None:
        await self._gateway.service_insert("demo_requests", payload=row)

    async def mark_notified(self, request_id: UUID) -> None:
        await self._gateway.service_update(
            "demo_requests",
            params={"id": f"eq.{request_id}"},
            payload={"notified": True},
        )
