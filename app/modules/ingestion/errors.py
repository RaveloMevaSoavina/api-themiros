from typing import Any

from app.core.errors import ApiError

# Erreurs passagères : le job est remis en file avec un délai croissant
# (spec §4.7 et §12.2 : « Retry avec backoff »).
RETRYABLE_CODES = frozenset(
    {
        "OCR_SERVICE_UNAVAILABLE",
        "EMBEDDING_SERVICE_UNAVAILABLE",
        "DOCUMENT_DOWNLOAD_FAILED",
        "DATABASE_UNAVAILABLE",
        "DATABASE_WRITE_FAILED",
    }
)

# Le job n'appartient plus à ce worker, ou un humain a tranché entre-temps :
# on s'arrête sans rien écrire.
ABANDON_CODES = frozenset({"LEASE_LOST", "HUMAN_DECISION_LOCKED"})


class IngestionError(ApiError):
    def __init__(
        self,
        *,
        code: str,
        detail: str,
        status_code: int = 400,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code, code=code, detail=detail, context=context
        )


def is_retryable(error: BaseException) -> bool:
    return isinstance(error, ApiError) and error.code in RETRYABLE_CODES


def is_abandon(error: BaseException) -> bool:
    return isinstance(error, ApiError) and error.code in ABANDON_CODES


def unsupported_format(filename: str) -> IngestionError:
    return IngestionError(
        code="UNSUPPORTED_FORMAT",
        detail="Seuls les fichiers PDF, DOCX et XLSX sont pris en charge",
        context={"filename": filename},
    )


def corrupted_file(filename: str) -> IngestionError:
    return IngestionError(
        code="CORRUPTED_FILE",
        detail="Le contenu du document est illisible ou corrompu",
        context={"filename": filename},
    )


def password_protected() -> IngestionError:
    return IngestionError(
        code="PASSWORD_PROTECTED_PDF",
        detail=(
            "Le PDF est protégé par un mot de passe : chargez une version "
            "sans protection"
        ),
    )


def no_exploitable_content() -> IngestionError:
    return IngestionError(
        code="NO_EXPLOITABLE_CONTENT",
        detail="Aucune page du document ne contient de texte exploitable",
    )


def ocr_unavailable() -> IngestionError:
    return IngestionError(
        status_code=503,
        code="OCR_SERVICE_UNAVAILABLE",
        detail="Le service OCR est indisponible",
    )


def embedding_unavailable() -> IngestionError:
    return IngestionError(
        status_code=503,
        code="EMBEDDING_SERVICE_UNAVAILABLE",
        detail="Le service d'embeddings est indisponible",
    )
