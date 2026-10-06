"""Indices textuels des règles complémentaires de pertinence (spec §7.4)."""

import re
from collections.abc import Iterable

from app.modules.ingestion.analysis.text import contains_phrase, count_phrase, normalize
from app.modules.ingestion.domain import RelevanceSignals

# Thématiques proposées à la création de l'espace, avec leurs formes FR, EN,
# PT et ES : un corpus mixte est le cas nominal (P-3).
THEME_TERMS: dict[str, tuple[str, ...]] = {
    "agriculture": (
        "agriculture",
        "agricole",
        "agricultural",
        "agricultura",
        "agricola",
        "farming",
        "agropastoral",
    ),
    "water": (
        "eau",
        "water",
        "agua",
        "hydraulique",
        "irrigation",
        "irrigacao",
        "riego",
        "hydrique",
    ),
    "nature": (
        "solutions fondees sur la nature",
        "nature based solutions",
        "solucoes baseadas na natureza",
        "soluciones basadas en la naturaleza",
        "ecosysteme",
        "ecosystem",
        "biodiversite",
        "biodiversity",
    ),
    "gender": ("genre", "gender", "genero", "femmes", "women", "mulheres", "mujeres"),
    "energy": ("energie", "energy", "energia", "renouvelable", "renewable"),
}

# Le malus « document technique hors sujet » ne s'applique que si les
# mots-clés sont saillants : en début de document ou répétés. Un programme
# qui cite une fois un « appel d'offres » n'est pas pénalisé.
OFF_TOPIC_HEAD_CHARS = 6_000
OFF_TOPIC_MIN_OCCURRENCES = 3
_ACRONYM = re.compile(r"^[A-Z0-9]{2,6}$")


def detect_signals(
    text: str,
    *,
    financier_terms: Iterable[str],
    themes: Iterable[str],
    off_topic_keywords: Iterable[str],
) -> RelevanceSignals:
    normalized = normalize(text)
    head = normalize(text[:OFF_TOPIC_HEAD_CHARS])

    financiers = sorted(
        {term for term in financier_terms if _mentions(text, normalized, term)}
    )
    themes_found = sorted(
        {
            theme
            for theme in themes
            if any(
                contains_phrase(normalized, term)
                for term in THEME_TERMS.get(theme, (theme,))
            )
        }
    )
    off_topic = sorted(
        {
            keyword
            for keyword in off_topic_keywords
            if contains_phrase(head, keyword)
            or count_phrase(normalized, keyword) >= OFF_TOPIC_MIN_OCCURRENCES
        }
    )
    return RelevanceSignals(
        financiers_found=tuple(financiers),
        themes_found=tuple(themes_found),
        off_topic_keywords_found=tuple(off_topic),
    )


def _mentions(text: str, normalized: str, term: str) -> bool:
    value = term.strip()
    if not value:
        return False
    # Un sigle (FIDA, UE, AFD) se cherche en majuscules et en mot entier :
    # « ue » ou « eu » minuscules sont des mots courants en FR et PT.
    if _ACRONYM.match(value):
        return (
            re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text)
            is not None
        )
    return contains_phrase(normalized, value)
