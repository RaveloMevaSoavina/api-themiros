"""Export d'un run terminé (E14, écran 22).

Écart au catalogue des API (§9) : le fichier est produit pendant la requête
et renvoyé directement, au lieu d'un 202 suivi d'un polling. Le rapport d'un
run tient en quelques secondes ; sans file ni stockage, la table `exports`
n'a pas lieu d'être. La traçabilité passe par le journal d'audit : chaque
export, réussi (`report_exported`) ou échoué (`report_export_failed`), y
entre avec son format, son run et son auteur.
"""

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.api.dependencies import CurrentUser
from app.core.errors import ApiError
from app.modules.exports.renderers import RENDERERS, filename
from app.modules.exports.report import build_report
from app.modules.exports.repository import ExportRepository
from app.modules.exports.schemas import ExportFormat, ExportRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExportFile:
    content: bytes
    media_type: str
    filename: str


class ExportService:
    def __init__(self, repository: ExportRepository) -> None:
        self._repository = repository

    async def export(
        self,
        run_id: UUID,
        export_format: ExportFormat,
        request: ExportRequest,
        user: CurrentUser,
    ) -> ExportFile:
        run = await self._repository.run(run_id, token=user.token)
        if run is None:
            raise ApiError(
                status_code=404, code="RUN_NOT_FOUND", detail="Run inexistant"
            )
        if run["status"] != "completed":
            raise ApiError(
                status_code=409,
                code="RUN_NOT_COMPLETED",
                detail="Seul un run terminé peut être exporté",
            )
        metadata = {
            "format": export_format,
            "locale": request.locale,
            "persona": request.persona,
            "engine_version": run.get("engine_version"),
            "corpus_seuil": run.get("corpus_seuil"),
            "exploratory": run.get("corpus_seuil") == "exploratoire",
            "validated": run.get("validated_at") is not None,
        }

        try:
            rows = await self._repository.rows(run, token=user.token)
            report = build_report(
                rows,
                locale=request.locale,
                persona=request.persona,
                exported_by=user.email or str(user.id),
                exported_at=datetime.now(UTC),
            )
            render, media_type, extension = RENDERERS[export_format]
            content = await asyncio.to_thread(render, report)
        except Exception as error:
            logger.exception("Export %s du run %s en échec", export_format, run_id)
            code = error.code if isinstance(error, ApiError) else "EXPORT_FAILED"
            # Un échec est journalisé au même titre qu'un succès (US-21.4).
            with suppress(ApiError):
                await self._record(
                    run, user, "report_export_failed", {**metadata, "error": code}
                )
            if isinstance(error, ApiError):
                raise
            raise ApiError(
                status_code=500,
                code="EXPORT_FAILED",
                detail="L'export n'a pas pu être généré",
            ) from error

        # Sans trace au journal, pas d'export (critère d'acceptation E14).
        await self._record(
            run,
            user,
            "report_exported",
            {**metadata, "notes": len(report.notes), "bytes": len(content)},
        )
        return ExportFile(
            content=content,
            media_type=media_type,
            filename=filename(report, extension),
        )

    async def _record(self, run, user: CurrentUser, operation: str, metadata) -> None:
        await self._repository.record(
            run,
            token=user.token,
            user_id=user.id,
            operation=operation,
            metadata=metadata,
        )
