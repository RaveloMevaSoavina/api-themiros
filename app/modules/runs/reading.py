"""Lecture des documents : une interprétation en paragraphe de chaque
segment, de chaque variable intermédiaire et de chaque pilier (prompt
read_evidences).

Déroulé, pour un run, une fois les couches A, B et C calculées et les
candidats de la restitution choisis :

1. `subjects` : pour chaque pilier, sa dernière version scorée, ses
   variables renseignées (et leurs meilleures preuves) et les intitulés de
   ses variables dont les documents ne parlent pas ;
2. `segments` : chaque segment que le run montre est lu une seule fois :
   preuves des variables retenues, preuves de la couche B, preuves des
   écarts et des constats de la restitution ;
3. `build_batches` : un seul appel en pratique. Un pilier part avec ses
   variables et leurs extraits ; les autres extraits suivent. Extraits
   découpés en phrases, identifiants courts (E1, V1, P1). Toutes les
   traductions (langues de l'interface) sont demandées dans le même appel
   (`tr`). Plusieurs appels seulement au-delà du plafond ;
4. `interpret` : chaque texte cite ses sources dans le texte (« ⟦1⟧ ») :
   la lecture d'un extrait, ses propres phrases ; l'interprétation d'une
   variable ou d'un pilier, ses extraits. Un texte trop court, ou sans
   source pour une variable ou un pilier, n'est pas publié.

Le projet évalué (`pj`) accompagne le message : les textes parlent de lui.

Sans IA (désactivée, en erreur, plafond de coût), rien n'est interprété :
les preuves s'affichent brutes, sans texte de substitution.

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from app.modules.evidences.passages import merge
from app.modules.frameworks.schemas import INTERFACE_LOCALES
from app.modules.runs.citations import (
    Excerpt,
    Reading,
    excerpt,
    read,
    reading_payload,
    written,
)
from app.modules.runs.domain import LayerAResults, PillarScore, VariableResult
from app.modules.runs.layer_b import LayerBResults, truncate
from app.modules.runs.project import ProjectContext

READING_VERSION = "reading/1.0.0"
PROMPT_CODE = "read_evidences"
PROMPT_VERSION = "v1.0"

CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_TOKENS = 1700
# Un paragraphe par texte, par langue.
OUTPUT_TOKENS_PER_SEGMENT = 280
OUTPUT_TOKENS_PER_VARIABLE = 280
OUTPUT_TOKENS_PER_PILLAR = 420
DESCRIPTION_CHARS = 200
# Le prompt (v1.0) demande 350 à 700 caractères hors sources pour un extrait
# ou une variable, 500 à 1 000 pour un pilier : ces plafonds ne coupent
# qu'un dépassement net ; sous les minimums, le texte n'est pas publié.
SEGMENT_CHARS = 850
SEGMENT_MIN_CHARS = 150
VARIABLE_CHARS = 850
VARIABLE_MIN_CHARS = 150
PILLAR_CHARS = 1200
PILLAR_MIN_CHARS = 250
# Plafonds de la base, marqueurs compris.
SEGMENT_MAX_CHARS = 1500
VARIABLE_MAX_CHARS = 1500
PILLAR_MAX_CHARS = 2000


@dataclass(frozen=True, slots=True)
class ReadingRules:
    # Preuves lues et citables par variable (les mieux placées).
    evidences_per_variable: int = 3
    excerpt_chars: int = 600
    max_input_tokens: int = 30000
    max_output_tokens: int = 16000

    def as_dict(self) -> dict[str, Any]:
        return {"engine_version": READING_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class VariableLabel:
    label: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ReadingInputs:
    locale: str
    pillar_names: Mapping[UUID, str]
    # (pilier, code) -> intitulé de la variable dans la langue du run.
    variables: Mapping[tuple[UUID, str], VariableLabel]
    project: ProjectContext = field(default_factory=ProjectContext)

    @staticmethod
    def labels(rows: Iterable[dict[str, Any]]) -> dict[tuple[UUID, str], VariableLabel]:
        return {
            (UUID(row["pillar_id"]), row["code"]): VariableLabel(
                row.get("label") or row["code"], row.get("description")
            )
            for row in rows
        }


@dataclass(slots=True)
class Subject:
    """Un pilier à interpréter : sa dernière version scorée, ses variables
    renseignées et leurs preuves lues."""

    score: PillarScore
    variables: list[VariableResult]
    # Intitulés des variables dont les documents ne parlent pas.
    missing: list[str]
    # Variable -> segments lus et citables.
    segments: dict[UUID, list[UUID]]

    def all_segments(self) -> list[UUID]:
        return list(
            dict.fromkeys(
                segment
                for variable in self.variables
                for segment in self.segments[variable.id]
            )
        )


@dataclass(frozen=True, slots=True)
class RawReadingTranslation:
    segments: Mapping[str, str] = field(default_factory=dict)
    variables: Mapping[str, str] = field(default_factory=dict)
    pillars: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawReading:
    """Réponse brute de l'IA (identifiants courts)."""

    segments: Mapping[str, str]
    variables: Mapping[str, str]
    pillars: Mapping[str, str]
    translations: Mapping[str, RawReadingTranslation] = field(default_factory=dict)


@dataclass(slots=True)
class ReadingBatch:
    """Un appel IA : contenu du message et correspondance des identifiants
    courts."""

    content: dict[str, Any] = field(default_factory=dict)
    excerpt_keys: dict[str, UUID] = field(default_factory=dict)
    excerpts: dict[str, Excerpt] = field(default_factory=dict)
    # Extraits à interpréter dans cet appel.
    to_read: list[str] = field(default_factory=list)
    variable_keys: dict[str, VariableResult] = field(default_factory=dict)
    pillar_keys: dict[str, PillarScore] = field(default_factory=dict)
    # Extraits citables par une variable ou un pilier : les siens.
    allowed: dict[str, frozenset[str]] = field(default_factory=dict)
    locales: tuple[str, ...] = ()
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0

    def message(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, separators=(",", ":"))

    def key_of(self, segment_id: UUID) -> str | None:
        return next(
            (
                key
                for key, segment in self.excerpt_keys.items()
                if segment == segment_id
            ),
            None,
        )


@dataclass(slots=True)
class ReadingResults:
    readings: dict[UUID, Reading] = field(default_factory=dict)
    variables: dict[UUID, Reading] = field(default_factory=dict)
    # (pilier, version) -> interprétation.
    pillars: dict[tuple[UUID, UUID], Reading] = field(default_factory=dict)
    ai: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "readings": len(self.readings),
            "variables": len(self.variables),
            "pillars": len(self.pillars),
            "ai_status": self.ai.get("status"),
            "ai_calls": self.ai.get("calls", 0),
        }

    def apply(self, payload: dict[str, Any]) -> None:
        """Ajoute au corps de `complete_run` la lecture de chaque segment
        (`readings`) et l'interprétation des variables et des piliers."""
        payload["readings"] = [
            {
                "segment_id": str(segment_id),
                "interpretation": reading.text,
                "translations": reading.translations,
                "interpretation_citations": [
                    source.as_dict() for source in reading.citations
                ],
                "spans": [list(span) for span in reading.spans] or None,
            }
            for segment_id, reading in self.readings.items()
        ]
        for item in payload.get("variables") or []:
            reading = self.variables.get(UUID(item["id"]))
            if reading is not None:
                item.update(reading_payload(reading))
        for item in payload.get("scores") or []:
            reading = self.pillars.get(
                (UUID(item["pillar_id"]), UUID(item["program_version_id"]))
            )
            if reading is not None:
                item.update(reading_payload(reading))


def subjects(
    layer_a: LayerAResults, inputs: ReadingInputs, rules: ReadingRules
) -> list[Subject]:
    """Piliers à interpréter, dans l'ordre des scores : la dernière version
    scorée de chacun (compute_layer_a trie les versions par order_index)."""
    latest: dict[UUID, PillarScore] = {}
    for score in layer_a.scores:
        if score.score is not None:
            latest[score.pillar_id] = score
    result: list[Subject] = []
    for pillar_id, score in latest.items():
        documented: list[VariableResult] = []
        missing: list[str] = []
        segments: dict[UUID, list[UUID]] = {}
        for variable in layer_a.variables:
            if (
                variable.pillar_id != pillar_id
                or variable.program_version_id != score.program_version_id
            ):
                continue
            label = inputs.variables.get((pillar_id, variable.variable_code))
            if variable.state == "non_renseigne" or not variable.evidences:
                missing.append(label.label if label else variable.variable_code)
                continue
            documented.append(variable)
            segments[variable.id] = list(
                dict.fromkeys(
                    match.segment_id
                    for match in variable.evidences[: rules.evidences_per_variable]
                )
            )
        if documented:
            result.append(Subject(score, documented, missing, segments))
    return result


def segments(
    pillars: Iterable[Subject],
    layer_b: LayerBResults | None,
    others: Iterable[Iterable[UUID]],
) -> list[UUID]:
    """Segments à lire, chacun une fois : preuves des variables retenues,
    de la couche B, puis des écarts et des constats (`others`)."""
    found: dict[UUID, None] = {}
    for subject in pillars:
        found.update(dict.fromkeys(subject.all_segments()))
    if layer_b is not None:
        for answer in layer_b.answers:
            if answer.status == "documentee":
                found.update(
                    dict.fromkeys(
                        candidate.segment_id for candidate in answer.evidences
                    )
                )
    for group in others:
        found.update(dict.fromkeys(group))
    return list(found)


def build_batches(
    inputs: ReadingInputs,
    pillars: list[Subject],
    to_read: list[UUID],
    texts: Mapping[UUID, str],
    rules: ReadingRules,
) -> list[ReadingBatch]:
    """Lots d'appels IA : un seul en pratique. Un pilier n'est jamais coupé
    de ses variables ni de leurs extraits ; un lot est fermé quand l'unité
    suivante ferait dépasser le plafond d'entrée ou de sortie."""
    locales = tuple(other for other in INTERFACE_LOCALES if other != inputs.locale)
    header: dict[str, Any] = {"lang": inputs.locale}
    if project := inputs.project.prompt():
        header["pj"] = project
    if locales:
        header["tr"] = list(locales)
    wanted = set(to_read)
    done: set[UUID] = set()

    units: list[tuple[Subject | None, list[UUID]]] = [
        (subject, [s for s in subject.all_segments() if s in texts])
        for subject in pillars
    ]
    in_units = {segment for _, unit in units for segment in unit}
    units += [(None, [segment]) for segment in to_read if segment not in in_units]

    batches: list[ReadingBatch] = []
    current = ReadingBatch(content={**header, "x": {}, "r": []}, locales=locales)
    for subject, unit in units:
        unit = [segment for segment in unit if segment in texts]
        if not unit:
            continue
        new = [segment for segment in unit if current.key_of(segment) is None]
        reads = [s for s in unit if s in wanted and s not in done]
        input_growth = (
            sum(min(len(texts[s]), rules.excerpt_chars) + 20 for s in new)
            + (300 * (1 + len(subject.variables)) if subject else 0)
        ) // CHARS_PER_TOKEN
        output_growth = (1 + len(locales)) * (
            OUTPUT_TOKENS_PER_SEGMENT * len(reads)
            + (
                OUTPUT_TOKENS_PER_VARIABLE * len(subject.variables)
                + OUTPUT_TOKENS_PER_PILLAR
                if subject
                else 0
            )
        )
        if (current.to_read or current.pillar_keys) and (
            current.estimated_input_tokens + input_growth > rules.max_input_tokens
            or current.max_output_tokens + output_growth > rules.max_output_tokens
        ):
            batches.append(current)
            current = ReadingBatch(
                content={**header, "x": {}, "r": []}, locales=locales
            )
        _add(current, inputs, subject, unit, texts, wanted, done, rules)
    if current.to_read or current.pillar_keys:
        batches.append(current)
    return batches


def _add(
    batch: ReadingBatch,
    inputs: ReadingInputs,
    subject: Subject | None,
    unit: list[UUID],
    texts: Mapping[UUID, str],
    wanted: set[UUID],
    done: set[UUID],
    rules: ReadingRules,
) -> None:
    keys: dict[UUID, str] = {}
    for segment_id in unit:
        key = batch.key_of(segment_id)
        if key is None:
            key = f"E{len(batch.excerpt_keys) + 1}"
            batch.excerpt_keys[key] = segment_id
            batch.excerpts[key] = excerpt(
                segment_id, texts[segment_id], rules.excerpt_chars
            )
            batch.content["x"][key] = list(batch.excerpts[key].sentences)
        keys[segment_id] = key
        if segment_id in wanted and segment_id not in done:
            done.add(segment_id)
            batch.to_read.append(key)
            batch.content["r"].append(key)
    if subject is not None:
        pillar_key = f"P{len(batch.pillar_keys) + 1}"
        variable_keys: list[str] = []
        for variable in subject.variables:
            variable_key = f"V{len(batch.variable_keys) + 1}"
            batch.variable_keys[variable_key] = variable
            own = [
                keys[segment]
                for segment in subject.segments[variable.id]
                if segment in keys
            ]
            batch.allowed[variable_key] = frozenset(own)
            label = inputs.variables.get((variable.pillar_id, variable.variable_code))
            entry: dict[str, Any] = {
                "k": variable_key,
                "p": inputs.pillar_names.get(variable.pillar_id, ""),
                "n": label.label if label else variable.variable_code,
                "s": variable.state,
                "dc": variable.documents_count,
                "e": own,
            }
            if label and label.description:
                entry["d"] = truncate(label.description, DESCRIPTION_CHARS)
            if variable.corpus_confidence is not None:
                entry["cf"] = round(variable.corpus_confidence, 2)
            batch.content.setdefault("v", []).append(entry)
            variable_keys.append(variable_key)
        score = subject.score
        batch.pillar_keys[pillar_key] = score
        own = list(dict.fromkeys(keys[segment] for segment in subject.all_segments()))
        batch.allowed[pillar_key] = frozenset(own)
        assert score.score is not None
        pillar: dict[str, Any] = {
            "k": pillar_key,
            "n": inputs.pillar_names.get(score.pillar_id, ""),
            "s": round(score.score, 1),
            "cf": round(score.corpus_confidence, 2),
            "v": variable_keys,
            "e": own,
        }
        if score.delta is not None:
            pillar["dl"] = round(score.delta, 1)
        if subject.missing:
            pillar["m"] = subject.missing
        batch.content.setdefault("p", []).append(pillar)
    batch.estimated_input_tokens = SYSTEM_PROMPT_TOKENS + (
        len(batch.message()) // CHARS_PER_TOKEN
    )
    batch.max_output_tokens = (1 + len(batch.locales)) * (
        OUTPUT_TOKENS_PER_SEGMENT * len(batch.to_read)
        + OUTPUT_TOKENS_PER_VARIABLE * len(batch.variable_keys)
        + OUTPUT_TOKENS_PER_PILLAR * len(batch.pillar_keys)
    )


def _normalized(items: Mapping[str, str]) -> dict[str, str]:
    return {key.strip().upper(): text for key, text in items.items()}


def interpret(batch: ReadingBatch, raw: RawReading, results: ReadingResults) -> None:
    """Applique une réponse IA : la lecture de chaque extrait cite ses
    propres phrases (à défaut, renvoie à l'extrait entier) ; l'interprétation
    d'une variable ou d'un pilier cite ses extraits, sans quoi elle n'est pas
    publiée. Un texte trop court n'est pas publié."""
    translations = {
        locale: raw.translations[locale]
        for locale in batch.locales
        if locale in raw.translations
    }
    segments_raw = _normalized(raw.segments)
    variables_raw = _normalized(raw.variables)
    pillars_raw = _normalized(raw.pillars)
    translated_segments = {
        locale: _normalized(translation.segments)
        for locale, translation in translations.items()
    }
    translated_variables = {
        locale: _normalized(translation.variables)
        for locale, translation in translations.items()
    }
    translated_pillars = {
        locale: _normalized(translation.pillars)
        for locale, translation in translations.items()
    }

    for key in batch.to_read:
        segment_id = batch.excerpt_keys[key]
        if segment_id in results.readings:
            continue
        reading = read(
            key,
            segments_raw.get(key),
            {locale: values.get(key) for locale, values in translated_segments.items()},
            batch.excerpts,
            SEGMENT_CHARS,
            SEGMENT_MAX_CHARS,
            min_chars=SEGMENT_MIN_CHARS,
        )
        if reading is None:
            continue
        reading.spans = tuple(
            merge([span for source in reading.citations for span in source.spans])
        )
        results.readings[segment_id] = reading

    for key, variable in batch.variable_keys.items():
        if variable.id in results.variables:
            continue
        reading = written(
            variables_raw.get(key),
            {
                locale: values.get(key)
                for locale, values in translated_variables.items()
            },
            batch.excerpts,
            batch.allowed.get(key, frozenset()),
            VARIABLE_CHARS,
            VARIABLE_MAX_CHARS,
            VARIABLE_MIN_CHARS,
        )
        if reading is not None:
            results.variables[variable.id] = reading

    for key, score in batch.pillar_keys.items():
        identity = (score.pillar_id, score.program_version_id)
        if identity in results.pillars:
            continue
        reading = written(
            pillars_raw.get(key),
            {locale: values.get(key) for locale, values in translated_pillars.items()},
            batch.excerpts,
            batch.allowed.get(key, frozenset()),
            PILLAR_CHARS,
            PILLAR_MAX_CHARS,
            PILLAR_MIN_CHARS,
        )
        if reading is not None:
            results.pillars[identity] = reading
