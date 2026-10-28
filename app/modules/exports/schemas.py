from typing import Literal

from pydantic import BaseModel

ExportFormat = Literal["pdf", "docx", "xlsx", "csv"]


class ExportRequest(BaseModel):
    """Langue des libellés et rôle de lecture du plan d'action."""

    locale: Literal["fr", "en"] = "fr"
    persona: Literal["decideur", "pmu", "analyste"] = "pmu"
