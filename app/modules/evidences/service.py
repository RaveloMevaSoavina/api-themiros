"""Document source d'une preuve, affiché dans le dashboard avec le passage
cité surligné.

Le fichier n'est lu avec la clé serveur qu'une fois la preuve lue au nom de
l'utilisateur : sans accès à la preuve, pas d'accès au document. Seul un PDF
se surligne ; un DOCX ou un XLSX s'ouvre tel quel depuis le stockage.
"""

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from app.api.dependencies import CurrentUser
from app.core.errors import ApiError
from app.modules.evidences.highlight import highlight
from app.modules.evidences.repository import EvidenceRepository

logger = logging.getLogger(__name__)

PDF_MEDIA_TYPE = "application/pdf"


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    content: bytes
    filename: str
    page: int | None
    highlighted: bool


class EvidenceService:
    def __init__(self, repository: EvidenceRepository) -> None:
        self._repository = repository

    async def document(self, evidence_id: UUID, user: CurrentUser) -> EvidenceDocument:
        evidence = await self._repository.evidence(evidence_id, token=user.token)
        if evidence is None:
            raise ApiError(
                status_code=404,
                code="EVIDENCE_NOT_FOUND",
                detail="Preuve inexistante",
            )
        document = evidence.get("documents")
        if not document:
            raise ApiError(
                status_code=404,
                code="EVIDENCE_DOCUMENT_NOT_FOUND",
                detail="Le document de la preuve a été supprimé",
            )
        if document.get("mime_type") != PDF_MEDIA_TYPE:
            raise ApiError(
                status_code=409,
                code="EVIDENCE_DOCUMENT_NOT_PDF",
                detail="Seul un document PDF peut être affiché surligné",
            )

        content = await self._repository.download(document["storage_path"])
        page = evidence.get("page_number")
        try:
            result = await asyncio.to_thread(
                highlight, content, evidence.get("excerpt") or "", page
            )
        except Exception:
            # Un PDF que MuPDF ne sait pas annoter reste lisible tel quel.
            logger.exception("Surlignage de la preuve %s en échec", evidence_id)
            return EvidenceDocument(content, document["original_filename"], page, False)
        return EvidenceDocument(
            result.content,
            document["original_filename"],
            result.page,
            result.highlighted,
        )
