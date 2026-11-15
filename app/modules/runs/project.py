"""Le projet évalué, tel que les prompts du run le reçoivent (`pj`).

Les résultats du moteur (piliers, critères, scores, notes) sont des repères
internes : sans savoir de quel projet il s'agit, l'IA rédige sur eux
(« le pilier X obtient 42/100 ») plutôt que sur le projet. Ce contexte
situe les extraits ; il n'est jamais une preuve.

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

from dataclasses import dataclass
from typing import Any

from app.modules.runs.citations import shorten

DESCRIPTION_CHARS = 600
NAME_CHARS = 160


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value else ""


@dataclass(frozen=True, slots=True)
class ProjectContext:
    name: str = ""
    # Nature : projet, programme ou politique.
    kind: str = ""
    description: str = ""
    country: str = ""
    financiers: tuple[str, ...] = ()
    themes: tuple[str, ...] = ()
    stage: str = ""
    start_year: int | None = None
    end_year: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "ProjectContext":
        row = row or {}

        def year(value: Any) -> int | None:
            return value if isinstance(value, int) else None

        return cls(
            name=shorten(_text(row.get("name")), NAME_CHARS),
            kind=_text(row.get("kind")),
            description=shorten(_text(row.get("description")), DESCRIPTION_CHARS),
            country=_text(row.get("country")),
            financiers=tuple(
                dict.fromkeys(
                    name
                    for item in row.get("financiers") or []
                    if (name := _text(item))
                )
            ),
            themes=tuple(
                dict.fromkeys(
                    theme for item in row.get("themes") or [] if (theme := _text(item))
                )
            ),
            stage=_text(row.get("stage")),
            start_year=year(row.get("start_year")),
            end_year=year(row.get("end_year")),
        )

    def period(self) -> str:
        if self.start_year and self.end_year:
            return f"{self.start_year}-{self.end_year}"
        return str(self.start_year or self.end_year or "")

    def prompt(self) -> dict[str, Any]:
        """Bloc `pj` des messages : clés courtes, champs vides omis."""
        fields: dict[str, Any] = {
            "n": self.name,
            "t": self.kind,
            "d": self.description,
            "pa": self.country,
            "f": list(self.financiers),
            "th": list(self.themes),
            "st": self.stage,
            "an": self.period(),
        }
        return {key: value for key, value in fields.items() if value}
