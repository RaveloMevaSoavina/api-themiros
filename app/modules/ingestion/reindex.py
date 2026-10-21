"""Revectorise les segments indexés avec un autre modèle que celui configuré.

Seuls les vecteurs changent : ni extraction, ni OCR, ni qualification. À
lancer avec la nouvelle configuration AVANT de redémarrer les workers : au
redémarrage, la classification revectorise les sondes des cadres
(`framework_needs_embedding`) et reclasse tout le corpus avec les nouveaux
vecteurs.

    python -m app.modules.ingestion.reindex
"""

import asyncio
import logging

import httpx

from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.analysis.embeddings import Embedder, build_embedder
from app.modules.ingestion.repository import vector_literal

logger = logging.getLogger(__name__)

PAGE_SIZE = 256


async def reindex(gateway: SupabaseGateway, embedder: Embedder) -> int:
    done = 0
    while True:
        rows = await gateway.service_select(
            "segment_embeddings",
            params={
                "select": "segment_id,workspace_id,segments(text)",
                "model": f"neq.{embedder.model}",
                "order": "segment_id",
                "limit": str(PAGE_SIZE),
            },
        )
        if not rows:
            return done
        vectors = await embedder.embed([row["segments"]["text"] for row in rows])
        await gateway.service_upsert(
            "segment_embeddings",
            on_conflict="segment_id",
            payload=[
                {
                    "segment_id": row["segment_id"],
                    "workspace_id": row["workspace_id"],
                    "embedding": vector_literal(vector),
                    "model": embedder.model,
                }
                for row, vector in zip(rows, vectors, strict=True)
            ],
        )
        done += len(rows)
        logger.info("%s segments revectorisés avec %s", done, embedder.model)


async def _main(settings: Settings) -> None:
    async with httpx.AsyncClient() as client:
        done = await reindex(
            SupabaseGateway(settings, client), build_embedder(settings)
        )
    logger.info("Réindexation terminée : %s segments", done)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(_main(get_settings()))
