"""Règles de décision de l'ingestion (spec §6.4, §7.4, §7.5, §7.6).

Fonctions pures : aucune dépendance au stockage, au réseau ou aux modèles.
"""

import gettext
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import pycountry

DocumentStatus = Literal["conforme", "a_verifier", "rejete"]

# Messages imposés, repris tels quels (Dossier de traduction produit E4).
COUNTRY_MISMATCH_MESSAGE = (
    "⚠ Ce document semble concerner {detected}, alors que ce workspace est "
    "configuré pour {target}. Souhaitez-vous l'ajouter quand même, vérifier le "
    "document, ou l'annuler ?"
)
OUT_OF_SCOPE_MESSAGE = (
    "⚠ Ce document ne semble pas lié au programme analysé dans ce workspace. "
    "Le moteur n'a pas identifié de contenu pertinent pour les piliers "
    "d'analyse définis. Souhaitez-vous l'ajouter quand même, vérifier, ou "
    "annuler ?"
)
PARTIAL_COVERAGE_MESSAGE = (
    "ℹ Ce document a été accepté, mais seules {n} pages sur {total} "
    "contiennent du contenu exploitable pour l'analyse. Les scores refléteront "
    "cette couverture partielle."
)
# Pas de message imposé pour ce cas (spec §6.4) : formulation à valider.
COUNTRY_UNDETECTED_MESSAGE = (
    "⚠ Le pays concerné par ce document n'a pas pu être détecté. "
    "Souhaitez-vous l'ajouter quand même, vérifier le document, ou l'annuler ?"
)


@dataclass(frozen=True, slots=True)
class IngestionRules:
    """Paramètres versionnés de la table `ingestion_settings` (RG-5.4)."""

    version: int = 1
    conformity_threshold: int = 70
    ambiguous_threshold: int = 40
    bonus_country: int = 10
    bonus_financier: int = 5
    bonus_theme: int = 5
    bonus_language: int = 5
    malus_off_topic: int = 20
    malus_other_country: int = 30
    language_confidence_threshold: float = 0.70
    exploitable_page_min_chars: int = 100
    off_topic_keywords: tuple[str, ...] = ()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "IngestionRules":
        return cls(
            version=int(row["version"]),
            conformity_threshold=int(row["conformity_threshold"]),
            ambiguous_threshold=int(row["ambiguous_threshold"]),
            bonus_country=int(row["bonus_country"]),
            bonus_financier=int(row["bonus_financier"]),
            bonus_theme=int(row["bonus_theme"]),
            bonus_language=int(row["bonus_language"]),
            malus_off_topic=int(row["malus_off_topic"]),
            malus_other_country=int(row["malus_other_country"]),
            language_confidence_threshold=float(row["language_confidence_threshold"]),
            exploitable_page_min_chars=int(row["exploitable_page_min_chars"]),
            off_topic_keywords=tuple(row.get("off_topic_keywords") or ()),
        )


@dataclass(frozen=True, slots=True)
class Detection:
    value: str | None
    confidence: float
    source: Literal["auto", "user"] = "auto"


@dataclass(frozen=True, slots=True)
class WorkspaceProfile:
    """Empreinte sémantique déclarée à la création de l'espace (E2)."""

    id: str
    kind: str
    target_country: str
    financiers: tuple[str, ...]
    themes: tuple[str, ...]
    expected_languages: tuple[str, ...]
    stage: str
    start_year: int | None
    end_year: int | None


@dataclass(frozen=True, slots=True)
class RelevanceSignals:
    """Indices textuels des règles complémentaires (spec §7.4)."""

    financiers_found: tuple[str, ...] = ()
    themes_found: tuple[str, ...] = ()
    off_topic_keywords_found: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RelevanceScore:
    semantic: int
    adjustments: dict[str, int] = field(default_factory=dict)

    @property
    def final(self) -> int:
        return max(0, min(100, self.semantic + sum(self.adjustments.values())))


@dataclass(frozen=True, slots=True)
class Decision:
    status: DocumentStatus
    reason_code: str | None
    reason_params: dict[str, Any]
    message: str | None


def score_relevance(
    *,
    semantic: int,
    profile: WorkspaceProfile,
    language: Detection,
    country: Detection,
    signals: RelevanceSignals,
    rules: IngestionRules,
) -> RelevanceScore:
    """Spec §7.4 : score final = min(100, max(0, sémantique + bonus − malus))."""
    adjustments: dict[str, int] = {}
    target = profile.target_country.upper()
    if country.value and target:
        if country.value == target:
            adjustments["country"] = rules.bonus_country
        else:
            adjustments["other_country"] = -rules.malus_other_country
    if signals.financiers_found:
        adjustments["financier"] = rules.bonus_financier
    if signals.themes_found:
        adjustments["theme"] = rules.bonus_theme
    if language.value and language.value in profile.expected_languages:
        adjustments["language"] = rules.bonus_language
    if signals.off_topic_keywords_found:
        adjustments["off_topic"] = -rules.malus_off_topic
    return RelevanceScore(semantic=max(0, min(100, semantic)), adjustments=adjustments)


def decide(
    *,
    score: int,
    profile: WorkspaceProfile,
    country: Detection,
    exploitable_pages: int,
    total_pages: int,
    rules: IngestionRules,
    locale: str = "fr",
) -> Decision:
    """Décision automatique (spec §6.4 et §7.5).

    Ordre : rejet sous le seuil ambigu, puis pays différent, puis pays non
    détecté, puis zone ambiguë, enfin conforme (avec couverture partielle).
    """
    target = profile.target_country.upper()
    other_country = bool(country.value and target and country.value != target)

    if score < rules.ambiguous_threshold:
        if other_country:
            return _country_mismatch("rejete", country.value, target, locale)
        return Decision("rejete", "out_of_scope", {}, OUT_OF_SCOPE_MESSAGE)
    if other_country:
        return _country_mismatch("a_verifier", country.value, target, locale)
    if not country.value:
        return Decision(
            "a_verifier", "country_undetected", {}, COUNTRY_UNDETECTED_MESSAGE
        )
    if score < rules.conformity_threshold:
        return Decision("a_verifier", "out_of_scope", {}, OUT_OF_SCOPE_MESSAGE)
    if exploitable_pages < total_pages:
        params = {"n": exploitable_pages, "total": total_pages}
        return Decision(
            "conforme",
            "partial_coverage",
            params,
            PARTIAL_COVERAGE_MESSAGE.format(**params),
        )
    return Decision("conforme", None, {}, None)


def is_eligible_for_index(status: str) -> bool:
    """RG-4.1 : seuls ces statuts alimentent le pipeline."""
    return status in {"conforme", "integre_decision_humaine"}


def fingerprint_text(profile: WorkspaceProfile) -> str:
    """Empreinte texte (spec §7.2), vectorisée pour la comparaison."""
    country = country_name(profile.target_country, "fr") or profile.target_country
    languages = ", ".join(code.upper() for code in profile.expected_languages)
    years = (
        f", années {profile.start_year}-{profile.end_year}"
        if profile.start_year and profile.end_year
        else ""
    )
    return (
        f"{profile.kind or 'Programme'} en {country}"
        f" financé par {', '.join(profile.financiers) or 'non précisé'}"
        f", thématiques {', '.join(profile.themes) or 'non précisées'}"
        f", stade {profile.stage or 'non précisé'}{years}"
        f", langues {languages}"
    )


def _country_mismatch(
    status: DocumentStatus, detected: str, target: str, locale: str
) -> Decision:
    params = {"detected": detected, "target": target}
    return Decision(
        status,
        "country_mismatch",
        params,
        COUNTRY_MISMATCH_MESSAGE.format(
            detected=country_name(detected, locale) or detected,
            target=country_name(target, locale) or target,
        ),
    )


@lru_cache(maxsize=8)
def _country_translation(locale: str) -> gettext.NullTranslations:
    try:
        return gettext.translation(
            "iso3166-1", pycountry.LOCALES_DIR, languages=[locale]
        )
    except OSError:
        return gettext.NullTranslations()


def country_name(code: str | None, locale: str = "fr") -> str | None:
    if not code:
        return None
    country = pycountry.countries.get(alpha_2=code.upper())
    if country is None:
        return None
    common_name = getattr(country, "common_name", None)
    # Le catalogue de traduction est indexé sur le nom officiel court.
    translated = _country_translation(locale).gettext(country.name)
    if translated == country.name and common_name:
        return common_name
    return translated
