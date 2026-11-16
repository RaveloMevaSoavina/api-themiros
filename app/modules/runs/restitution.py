"""Restitution : synthèse et plan d'action par persona (Algorithmes de
calcul §8, Spécification des prompts IA §8 et §9).

Déroulé, pour un run :

1. `select` : règles déterministes sur les résultats déjà calculés du run.

   - progrès : piliers dont la dernière version scorée est « renforcée »,
     par évolution décroissante ;
   - faiblesses : piliers sous le seuil de score et critères conclus sous le
     seuil de note, ceux qui portent une alerte majeure d'abord, puis du
     plus faible au moins faible ;
   - zones à vérifier : alertes publiées, majeures d'abord ;
   - limites du corpus : critères non conclus et confiance moyenne ; la
     synthèse ouvre par elles quand plusieurs critères sont non conclus ou
     que le corpus est pauvre (RG-2).

   Trois au plus par rubrique : on affiche ce qui existe (§8.6). Un
   candidat sans preuve n'est pas retenu (GF-1). Les origines des actions
   sont les faiblesses, les alertes et les critères non conclus documentés
   en partie ;
2. `build_batch` : tous les candidats en un seul appel IA (prompt
   formulate_restitution), extraits dédoublonnés et découpés en phrases,
   identifiants courts (P1, W1, V1, N1, E1). Le volume est borné par
   construction (neuf puces, six actions) : jamais plus d'un appel ;
3. `interpret` : l'IA rédige chaque constat en paragraphe, la conclusion
   prudente et chaque action avec ses trois projections persona (l'action
   et sa raison, un paragraphe), en citant dans le texte les phrases des
   extraits proposés pour elle (`citations`). Un constat trop court garde
   son libellé factuel ; une action sans ses trois projections, ou dont une
   raison est trop courte, n'est pas publiée (RG-4). Les extraits sont lus
   par l'étape de lecture (`reading`), une fois par run.

Le projet évalué (`pj`) accompagne le message : tous ces textes parlent
de lui, pas des piliers, des critères ni de leurs scores, affichés à côté.

Sans IA (désactivée, en erreur, plafond de coût), les puces sont publiées
avec un libellé factuel, sans conclusion ni plan d'action : la synthèse
informe sans rien interpréter.

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from app.modules.evidences.passages import Span
from app.modules.frameworks.schemas import INTERFACE_LOCALES
from app.modules.runs.citations import (
    Citations,
    Excerpt,
    Source,
    cited,
    compact,
    cut,
    excerpt,
    fit,
    long_enough,
    references,
)
from app.modules.runs.citations import truncate as truncate_cited
from app.modules.runs.domain import LayerAResults, PillarScore
from app.modules.runs.layer_b import LayerBResults, truncate
from app.modules.runs.layer_c import Alert, LayerCInputs, ordered_segments
from app.modules.runs.project import ProjectContext

RESTITUTION_VERSION = "restitution/1.6.0"
PROMPT_CODE = "formulate_restitution"
# v1.2 : le même appel traduit aussi la restitution dans les autres langues
# de l'interface (`tr`). v1.3 : extraits en phrases, sources citées dans le
# texte, interprétation de chaque extrait (`x`). v1.4 : le projet évalué
# (`pj`), sujet de tous les textes ; interprétations sourcées. v1.5 :
# constats et raisons des actions en paragraphes ; les extraits sont lus
# par l'étape de lecture.
PROMPT_VERSION = "v1.5"

Section = Literal["progres", "faiblesse", "a_verifier"]
# Rédaction de la synthèse : faite, inutile (aucun constat), IA désactivée,
# ou en échec (prompt introuvable, erreur, réponse inexploitable).
Status = Literal["completed", "not_needed", "disabled", "failed"]
OriginType = Literal["pillar", "criterion", "alert"]
PERSONAS = ("decideur", "pmu", "analyste")
# Clés courtes des personas dans la réponse de l'IA.
PERSONA_KEYS = {"decideur": "dec", "pmu": "pmu", "analyste": "ana"}

# Mêmes estimations que les couches B et C (~3 caractères par token).
CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_TOKENS = 1500
OUTPUT_TOKENS_PER_ITEM = 300
OUTPUT_TOKENS_PER_ACTION = 1300
OUTPUT_TOKENS_CONCLUSION = 650
# Le prompt (v1.5) demande 1 800 caractères pour la conclusion, 350 à 800
# pour un constat, 450 pour une action et 300 à 700 pour sa raison : ces
# plafonds ne coupent qu'un dépassement net. Sous les minimums, une phrase
# de circonstance n'est pas publiée.
ITEM_CHARS = 900
ITEM_MIN_CHARS = 150
CONCLUSION_CHARS = 2400
PROJECTION_CHARS = 850
REASON_MIN_CHARS = 120
# Plafonds de la base, marqueurs de sources compris.
ITEM_MAX_CHARS = 1500
CONCLUSION_MAX_CHARS = 3000
PROJECTION_MAX_CHARS = 1500

# « Recommandée parce que » est ajouté à l'affichage.
BECAUSE = re.compile(
    r"^\s*(recommand[ée]e?s?\s+parce\s+que|recommended\s+because)\s*:?\s*", re.I
)


@dataclass(frozen=True, slots=True)
class RestitutionRules:
    items_per_section: int = 3
    # Faiblesses relatives (§8.4) : les plus faibles sous 80 % du maximum,
    # soit sous « excellent » (couche A) et sous 2,4/3 (couche B).
    weak_score: float = 80
    weak_note: float = 2.4
    poor_confidence: float = 0.5
    limits_from: int = 2
    evidences_per_item: int = 3
    # Assez de phrases pour que l'IA cite la bonne.
    excerpt_chars: int = 600
    max_actions: int = 6

    def as_dict(self) -> dict[str, Any]:
        return {"engine_version": RESTITUTION_VERSION, **asdict(self)}


@dataclass(slots=True)
class Finding:
    """Candidat de la synthèse (P, W, V) ou origine d'action seule (N)."""

    id: UUID
    key: str
    section: Section | None
    origin: OriginType
    text: str
    # Faits soumis à l'IA, sans les extraits.
    facts: dict[str, Any]
    segment_ids: list[UUID]
    pillar_id: UUID | None = None
    program_version_id: UUID | None = None
    criterion_id: UUID | None = None
    alert_id: UUID | None = None
    # Constat traduit, par langue de l'interface (autre que celle du run).
    translations: dict[str, str] = field(default_factory=dict)
    # Sources numérotées du constat et de ses traductions (« ⟦1⟧ »).
    citations: list[Source] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Projection:
    persona: str
    text: str
    justification: str
    # Langue -> {"text", "justification"}.
    translations: dict[str, dict[str, str]] = field(default_factory=dict)
    citations: list[Source] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "text": self.text,
            "justification": self.justification,
            "translations": self.translations,
            "citations": [source.as_dict() for source in self.citations],
        }


@dataclass(slots=True)
class Action:
    id: UUID
    finding: Finding
    segment_ids: list[UUID]
    projections: list[Projection]


@dataclass(frozen=True, slots=True)
class Limits:
    not_concluded: int
    criteria: int
    confidence: float
    leads: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RawProjection:
    text: str
    justification: str


@dataclass(frozen=True, slots=True)
class RawItem:
    key: str
    text: str
    excerpts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawAction:
    key: str
    excerpts: tuple[str, ...]
    # Persona (decideur, pmu, analyste) -> formulation.
    projections: Mapping[str, RawProjection]


@dataclass(frozen=True, slots=True)
class RawTranslation:
    """Traduction de la réponse dans une langue : constats et actions par
    identifiant court."""

    items: Mapping[str, str]
    conclusion: str
    # Identifiant -> persona -> formulation.
    actions: Mapping[str, Mapping[str, RawProjection]]


@dataclass(frozen=True, slots=True)
class RawRestitution:
    """Réponse brute de l'IA (identifiants courts)."""

    items: tuple[RawItem, ...]
    conclusion: str
    actions: tuple[RawAction, ...]
    # Langue -> traduction ; seules les langues demandées (`tr`) comptent.
    translations: Mapping[str, RawTranslation] = field(default_factory=dict)


@dataclass(slots=True)
class RestitutionBatch:
    """L'appel IA : contenu du message et correspondance des identifiants
    courts."""

    content: dict[str, Any] = field(default_factory=dict)
    findings: dict[str, Finding] = field(default_factory=dict)
    excerpt_keys: dict[str, UUID] = field(default_factory=dict)
    excerpts: dict[str, Excerpt] = field(default_factory=dict)
    # Extraits proposés pour chaque candidat : seuls citables (GF-1).
    allowed: dict[str, frozenset[str]] = field(default_factory=dict)
    origins: tuple[str, ...] = ()
    # Langues de traduction demandées (`tr`), hors langue de rédaction.
    locales: tuple[str, ...] = ()
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0

    def message(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class Restitution:
    limits: Limits
    findings: list[Finding]
    actions: list[Action] = field(default_factory=list)
    conclusion: str | None = None
    qualified: bool = False
    ai: dict[str, Any] = field(default_factory=dict)
    # Langue de rédaction (celle de l'approche) et conclusion traduite.
    locale: str | None = None
    conclusion_translations: dict[str, str] = field(default_factory=dict)
    conclusion_citations: list[Source] = field(default_factory=list)

    @property
    def status(self) -> Status:
        if self.qualified:
            return "completed"
        if self.ai.get("status") == "not_needed":
            return "not_needed"
        if "AI_DISABLED" in self.ai.get("errors", ()):
            return "disabled"
        return "failed"

    def items(self, section: Section) -> list[Finding]:
        return [finding for finding in self.findings if finding.section == section]

    def origins(self) -> list[Finding]:
        """Origines des actions, dans l'ordre du plan."""
        return [
            finding for finding in self.findings if finding.key[0] in ("W", "V", "N")
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "progresses": len(self.items("progres")),
            "weaknesses": len(self.items("faiblesse")),
            "to_verify": len(self.items("a_verifier")),
            "actions": len(self.actions),
            "limits_lead": self.limits.leads,
            "qualified": self.qualified,
            "status": self.status,
            "ai_status": self.ai.get("status"),
            "ai_calls": self.ai.get("calls", 0),
        }

    def payload(self) -> dict[str, Any]:
        """Partie `restitution` du corps de `complete_run`."""

        def origin(finding: Finding) -> dict[str, str | None]:
            return {
                name: str(value) if value else None
                for name, value in (
                    ("pillar_id", finding.pillar_id),
                    ("program_version_id", finding.program_version_id),
                    ("criterion_id", finding.criterion_id),
                    ("alert_id", finding.alert_id),
                )
            }

        sections: tuple[Section, ...] = ("progres", "faiblesse", "a_verifier")
        return {
            "synthesis": {
                "conclusion": self.conclusion,
                "limits": self.limits.as_dict(),
                "qualified": self.qualified,
                "status": self.status,
                "locale": self.locale,
                "translations": {
                    locale: {"conclusion": conclusion}
                    for locale, conclusion in self.conclusion_translations.items()
                },
                "citations": [source.as_dict() for source in self.conclusion_citations],
            },
            "items": [
                {
                    "id": str(finding.id),
                    "section": section,
                    "position": position,
                    "text": finding.text,
                    "translations": finding.translations,
                    **origin(finding),
                    "segment_ids": [str(segment) for segment in finding.segment_ids],
                    "citations": [source.as_dict() for source in finding.citations],
                }
                for section in sections
                for position, finding in enumerate(self.items(section), start=1)
            ],
            "actions": [
                {
                    "id": str(action.id),
                    "position": position,
                    "origin_type": action.finding.origin,
                    **origin(action.finding),
                    "segment_ids": [str(segment) for segment in action.segment_ids],
                    "projections": [
                        projection.as_dict() for projection in action.projections
                    ],
                }
                for position, action in enumerate(self.actions, start=1)
            ],
        }


LABELS = {
    "fr": {
        "progress": (
            "{pillar} : {score}/100, +{delta} points depuis la version précédente."
        ),
        "weak_pillar": "{pillar} : {score}/100 (confiance corpus {confidence} %).",
        "weak_criterion": "{criterion} : note {note}/3.",
        "not_concluded": (
            "{criterion} : non conclu "
            "({documented}/{total} sous-questions documentées)."
        ),
    },
    "en": {
        "progress": (
            "{pillar}: {score}/100, +{delta} points since the previous version."
        ),
        "weak_pillar": "{pillar}: {score}/100 (corpus confidence {confidence}%).",
        "weak_criterion": "{criterion}: rated {note}/3.",
        "not_concluded": (
            "{criterion}: not concluded "
            "({documented}/{total} sub-questions documented)."
        ),
    },
}


def label(locale: str, kind: str, **values: Any) -> str:
    texts = LABELS.get(locale, LABELS["en"])
    return truncate(texts[kind].format(**values), ITEM_CHARS)


def latest_scores(
    layer_a: LayerAResults,
) -> dict[UUID, tuple[PillarScore, PillarScore | None]]:
    """Par pilier : sa dernière version, et sa dernière version scorée
    (compute_layer_a trie les versions par order_index)."""
    by_pillar: dict[UUID, list[PillarScore]] = defaultdict(list)
    for score in layer_a.scores:
        by_pillar[score.pillar_id].append(score)
    result: dict[UUID, tuple[PillarScore, PillarScore | None]] = {}
    for pillar_id, scores in by_pillar.items():
        scored = [score for score in scores if score.score is not None]
        result[pillar_id] = (scores[-1], scored[-1] if scored else None)
    return result


def corpus_limits(
    layer_a: LayerAResults, layer_b: LayerBResults | None, rules: RestitutionRules
) -> Limits:
    notes = layer_b.notes if layer_b is not None else []
    not_concluded = sum(note.status == "non_conclu" for note in notes)
    latest = [last for last, _ in latest_scores(layer_a).values()]
    confidence = (
        round(sum(score.corpus_confidence for score in latest) / len(latest), 2)
        if latest
        else 0.0
    )
    return Limits(
        not_concluded=not_concluded,
        criteria=len(notes),
        confidence=confidence,
        leads=not_concluded >= rules.limits_from or confidence < rules.poor_confidence,
    )


def select(
    inputs: LayerCInputs,
    layer_a: LayerAResults,
    layer_b: LayerBResults | None,
    alerts: Iterable[Alert],
    rules: RestitutionRules,
    new_id: Callable[[], UUID] = uuid4,
) -> Restitution:
    alerts = list(alerts)
    limit = rules.evidences_per_item
    criterion_names = {link.criterion_id: link.name for link in inputs.links}

    # Preuves de la couche A par pilier et version, de la couche B par critère.
    pillar_matches: dict[tuple[UUID, UUID], list[tuple[UUID, float]]] = defaultdict(
        list
    )
    for variable in layer_a.variables:
        pillar_matches[(variable.pillar_id, variable.program_version_id)].extend(
            (match.segment_id, match.similarity) for match in variable.evidences
        )
    criterion_matches: dict[UUID, list[tuple[UUID, float]]] = defaultdict(list)
    if layer_b is not None:
        for answer in layer_b.answers:
            if answer.status == "documentee" and answer.criterion_id is not None:
                criterion_matches[answer.criterion_id].extend(
                    (candidate.segment_id, candidate.similarity)
                    for candidate in answer.evidences
                )

    def pillar_segments(score: PillarScore) -> list[UUID]:
        return ordered_segments(
            pillar_matches.get((score.pillar_id, score.program_version_id), ())
        )[:limit]

    def criterion_segments(criterion_id: UUID) -> list[UUID]:
        return ordered_segments(criterion_matches.get(criterion_id, ()))[:limit]

    scored = [
        score for _, score in latest_scores(layer_a).values() if score is not None
    ]

    # Progrès : dernière version scorée « renforcée ».
    progresses: list[Finding] = []
    for score in sorted(
        (
            score
            for score in scored
            if score.evolution_label == "renforce" and score.delta is not None
        ),
        key=lambda score: (-(score.delta or 0), str(score.pillar_id)),
    ):
        segments = pillar_segments(score)
        if not segments:
            continue
        name = inputs.pillar_names.get(score.pillar_id, "")
        assert score.score is not None and score.delta is not None
        progresses.append(
            Finding(
                id=new_id(),
                key=f"P{len(progresses) + 1}",
                section="progres",
                origin="pillar",
                text=label(
                    inputs.locale,
                    "progress",
                    pillar=name,
                    score=f"{score.score:.0f}",
                    delta=f"{score.delta:.0f}",
                ),
                facts={
                    "p": name,
                    "s": round(score.score, 1),
                    "d": round(score.delta, 1),
                    "cf": round(score.corpus_confidence, 2),
                },
                segment_ids=segments,
                pillar_id=score.pillar_id,
                program_version_id=score.program_version_id,
            )
        )
        if len(progresses) == rules.items_per_section:
            break

    # Faiblesses : piliers et critères faibles, alerte majeure d'abord.
    major_pillars = {alert.pillar_id for alert in alerts if alert.severity == "majeure"}
    major_criteria = {
        alert.criterion_id
        for alert in alerts
        if alert.severity == "majeure" and alert.criterion_id is not None
    }
    weak: list[tuple[tuple[int, float, str], Finding]] = []
    for score in scored:
        assert score.score is not None
        if score.score >= rules.weak_score:
            continue
        segments = pillar_segments(score)
        if not segments:
            continue
        name = inputs.pillar_names.get(score.pillar_id, "")
        weak.append(
            (
                (
                    0 if score.pillar_id in major_pillars else 1,
                    score.score / 100,
                    str(score.pillar_id),
                ),
                Finding(
                    id=new_id(),
                    key="",
                    section="faiblesse",
                    origin="pillar",
                    text=label(
                        inputs.locale,
                        "weak_pillar",
                        pillar=name,
                        score=f"{score.score:.0f}",
                        confidence=f"{score.corpus_confidence * 100:.0f}",
                    ),
                    facts={
                        "p": name,
                        "s": round(score.score, 1),
                        "cf": round(score.corpus_confidence, 2),
                    },
                    segment_ids=segments,
                    pillar_id=score.pillar_id,
                    program_version_id=score.program_version_id,
                ),
            )
        )
    notes = layer_b.notes if layer_b is not None else []
    for note in notes:
        if note.status != "conclu" or note.note is None or note.note >= rules.weak_note:
            continue
        name = criterion_names.get(note.criterion_id)
        segments = criterion_segments(note.criterion_id)
        if not name or not segments:
            continue
        weak.append(
            (
                (
                    0 if note.criterion_id in major_criteria else 1,
                    note.note / 3,
                    str(note.criterion_id),
                ),
                Finding(
                    id=new_id(),
                    key="",
                    section="faiblesse",
                    origin="criterion",
                    text=label(
                        inputs.locale,
                        "weak_criterion",
                        criterion=name,
                        note=f"{note.note:.1f}",
                    ),
                    facts={"c": name, "n": round(note.note, 1)},
                    segment_ids=segments,
                    criterion_id=note.criterion_id,
                ),
            )
        )
    weaknesses = [finding for _, finding in sorted(weak, key=lambda item: item[0])][
        : rules.items_per_section
    ]
    for index, finding in enumerate(weaknesses, start=1):
        finding.key = f"W{index}"

    # Zones à vérifier : alertes publiées, majeures d'abord (tri stable).
    to_verify: list[Finding] = []
    for alert in sorted(alerts, key=lambda alert: alert.severity != "majeure"):
        if not alert.segment_ids:
            continue
        facts: dict[str, Any] = {
            "g": alert.severity,
            "m": alert.message,
            "p": alert.pillar_name,
        }
        if alert.criterion_name:
            facts["c"] = alert.criterion_name
        to_verify.append(
            Finding(
                id=new_id(),
                key=f"V{len(to_verify) + 1}",
                section="a_verifier",
                origin="alert",
                text=truncate(alert.message, ITEM_CHARS),
                facts=facts,
                segment_ids=alert.segment_ids[:limit],
                alert_id=alert.id,
            )
        )
        if len(to_verify) == rules.items_per_section:
            break

    # Critères non conclus documentés en partie : origines d'actions seules.
    remaining = rules.max_actions - len(weaknesses) - len(to_verify)
    partial: list[Finding] = []
    for note in sorted(
        (note for note in notes if note.status == "non_conclu"),
        key=lambda note: (-note.documented, str(note.criterion_id)),
    ):
        if len(partial) >= remaining:
            break
        name = criterion_names.get(note.criterion_id)
        segments = criterion_segments(note.criterion_id)
        if not name or not segments:
            continue
        partial.append(
            Finding(
                id=new_id(),
                key=f"N{len(partial) + 1}",
                section=None,
                origin="criterion",
                text=label(
                    inputs.locale,
                    "not_concluded",
                    criterion=name,
                    documented=note.documented,
                    total=note.total,
                ),
                facts={"c": name, "dq": note.documented, "tq": note.total},
                segment_ids=segments,
                criterion_id=note.criterion_id,
            )
        )

    return Restitution(
        limits=corpus_limits(layer_a, layer_b, rules),
        findings=progresses + weaknesses + to_verify + partial,
        locale=inputs.locale,
    )


def build_batch(
    locale: str,
    restitution: Restitution,
    texts: Mapping[UUID, str],
    rules: RestitutionRules,
    project: ProjectContext | None = None,
) -> RestitutionBatch | None:
    """L'appel IA, ou `None` quand il n'y a rien à rédiger."""
    if not restitution.findings:
        return None
    limits = restitution.limits
    targets = tuple(other for other in INTERFACE_LOCALES if other != locale)
    header: dict[str, Any] = {"lang": locale}
    if project is not None and (context := project.prompt()):
        header["pj"] = context
    batch = RestitutionBatch(
        locales=targets,
        content={
            **header,
            "l": {
                "nc": limits.not_concluded,
                "tc": limits.criteria,
                "cf": limits.confidence,
                "ld": limits.leads,
            },
            "x": {},
        },
    )
    key_of_segment: dict[UUID, str] = {}
    for finding in restitution.findings:
        keys: list[str] = []
        for segment_id in finding.segment_ids:
            if segment_id not in texts:
                continue
            excerpt_key = key_of_segment.get(segment_id)
            if excerpt_key is None:
                excerpt_key = f"E{len(key_of_segment) + 1}"
                key_of_segment[segment_id] = excerpt_key
                batch.excerpt_keys[excerpt_key] = segment_id
                batch.excerpts[excerpt_key] = excerpt(
                    segment_id, texts[segment_id], rules.excerpt_chars
                )
                batch.content["x"][excerpt_key] = list(
                    batch.excerpts[excerpt_key].sentences
                )
            keys.append(excerpt_key)
        batch.findings[finding.key] = finding
        batch.allowed[finding.key] = frozenset(keys)
        group = finding.key[0].lower()
        batch.content.setdefault(group, []).append(
            {"k": finding.key, **finding.facts, "e": keys}
        )
    batch.origins = tuple(finding.key for finding in restitution.origins())[
        : rules.max_actions
    ]
    if batch.origins:
        batch.content["o"] = list(batch.origins)
    if targets:
        batch.content["tr"] = list(targets)
    items = sum(finding.section is not None for finding in restitution.findings)
    batch.estimated_input_tokens = SYSTEM_PROMPT_TOKENS + (
        len(batch.message()) // CHARS_PER_TOKEN
    )
    # Chaque traduction réécrit constats, conclusion et actions.
    batch.max_output_tokens = (1 + len(targets)) * (
        OUTPUT_TOKENS_CONCLUSION
        + OUTPUT_TOKENS_PER_ITEM * items
        + OUTPUT_TOKENS_PER_ACTION * len(batch.origins)
    )
    return batch


def _cited(
    batch: RestitutionBatch,
    key: str,
    excerpts: Iterable[str],
    spans: dict[str, list[Span]],
) -> list[UUID]:
    """Extraits cités parmi ceux proposés pour le candidat ; leurs phrases
    citées s'ajoutent à `spans`."""
    found = cited(excerpts, batch.excerpts, batch.allowed.get(key, frozenset()))
    for excerpt_key, cited_spans in found.items():
        spans.setdefault(excerpt_key, []).extend(cited_spans)
    return [batch.excerpt_keys[excerpt_key] for excerpt_key in found]


def _reason(text: str) -> str:
    return truncate_cited(BECAUSE.sub("", text, count=1), PROJECTION_CHARS)


def _narrative(text: str, limit: int) -> str:
    """Conclusion en paragraphes : espaces normalisés dans chacun, lignes
    vides gardées entre eux, puis coupe au plafond comme `truncate`."""
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    return cut("\n\n".join(paragraphs), limit)


def _projection(
    projection: RawProjection | None,
    citations: Citations,
    allowed: frozenset[str],
) -> dict[str, str] | None:
    """Formulation exploitable, ses sources numérotées dans `citations`, ou
    `None` (texte ou raison manquant)."""
    if projection is None or not projection.text.strip():
        return None
    reason = _reason(citations.cite(projection.justification, allowed))
    if not long_enough(reason, REASON_MIN_CHARS):
        return None
    return {
        "text": truncate_cited(
            citations.cite(projection.text, allowed), PROJECTION_CHARS
        ),
        "justification": reason,
    }


def _written(
    text: str | None,
    citations: Citations,
    allowed: frozenset[str] | None,
    limit: int,
) -> str | None:
    """Texte de l'IA, ses sources numérotées, coupé à `limit` caractères
    hors marqueurs ; `None` s'il est vide."""
    if not text or not text.strip():
        return None
    return truncate_cited(citations.cite(text, allowed), limit)


def interpret(
    restitution: Restitution,
    batch: RestitutionBatch,
    raw: RawRestitution,
    new_id: Callable[[], UUID] = uuid4,
) -> None:
    """Applique la réponse de l'IA. Une puce sans texte garde son libellé
    factuel ; sans extrait valable, elle garde ses preuves. Une action sans
    ses trois projections, ou hors des origines proposées, est ignorée.

    Sources : une puce et une action ne citent que les extraits proposés
    pour elles (leur liste `e` et les marqueurs de leurs textes) ; la
    conclusion, tous les extraits de l'appel. Chaque texte numérote ses
    sources, les mêmes dans toutes ses traductions.

    Traductions : seules les langues demandées comptent, et seulement pour
    ce qui a été rédigé dans la langue du run. Une action n'est traduite
    qu'avec ses trois projections ; ce qui manque s'affiche dans la langue
    du run."""
    translations = {
        locale: raw.translations[locale]
        for locale in batch.locales
        if locale in raw.translations
    }
    # Phrases citées par extrait, pour la lecture des preuves.
    spans: dict[str, list[Span]] = {}
    seen: set[str] = set()
    for item in raw.items:
        finding = batch.findings.get(item.key)
        if finding is None or finding.section is None or item.key in seen:
            continue
        seen.add(item.key)
        allowed = batch.allowed.get(item.key, frozenset())
        citations = Citations(batch.excerpts)
        text = _written(item.text, citations, allowed, ITEM_CHARS)
        if text and long_enough(text, ITEM_MIN_CHARS):
            translated = {
                locale: _written(
                    translation.items.get(item.key), citations, allowed, ITEM_CHARS
                )
                for locale, translation in translations.items()
            }
            locales = [
                locale
                for locale, value in translated.items()
                if long_enough(value, ITEM_MIN_CHARS)
            ]
            texts, finding.citations = compact(
                [text, *(translated[locale] for locale in locales)],
                citations.sources,
            )
            finding.text = fit(texts[0] or "", ITEM_MAX_CHARS)
            for locale, value in zip(locales, texts[1:], strict=True):
                finding.translations[locale] = fit(value or "", ITEM_MAX_CHARS)
        found = _cited(batch, item.key, [*item.excerpts, *references(item.text)], spans)
        if found:
            finding.segment_ids = found

    # Conclusion en paragraphes : `_narrative` garde les lignes vides.
    citations = Citations(batch.excerpts)

    def narrative(text: str | None) -> str | None:
        if not text or not text.strip():
            return None
        return _narrative(citations.cite(text), CONCLUSION_CHARS) or None

    conclusion = narrative(raw.conclusion)
    if conclusion:
        translated = {
            locale: narrative(translation.conclusion)
            for locale, translation in translations.items()
        }
        locales = [locale for locale, value in translated.items() if value]
        texts, restitution.conclusion_citations = compact(
            [conclusion, *(translated[locale] for locale in locales)],
            citations.sources,
        )
        restitution.conclusion = fit(texts[0] or "", CONCLUSION_MAX_CHARS)
        for locale, value in zip(locales, texts[1:], strict=True):
            restitution.conclusion_translations[locale] = fit(
                value or "", CONCLUSION_MAX_CHARS
            )
        for source in restitution.conclusion_citations:
            key = next(
                key
                for key, item in batch.excerpts.items()
                if item.segment_id == source.segment_id
            )
            spans.setdefault(key, []).extend(source.spans)
    else:
        restitution.conclusion = None

    actions: dict[str, Action] = {}
    for raw_action in raw.actions:
        finding = batch.findings.get(raw_action.key)
        if (
            finding is None
            or raw_action.key not in batch.origins
            or raw_action.key in actions
        ):
            continue
        allowed = batch.allowed.get(raw_action.key, frozenset())
        by_persona = {persona: Citations(batch.excerpts) for persona in PERSONAS}
        written = {
            persona: _projection(
                raw_action.projections.get(persona), by_persona[persona], allowed
            )
            for persona in PERSONAS
        }
        if not all(written.values()):
            continue
        translated: dict[str, dict[str, dict[str, str] | None]] = {}
        for locale, translation in translations.items():
            raw_versions = translation.actions.get(raw_action.key) or {}
            versions = {
                persona: _projection(
                    raw_versions.get(persona), by_persona[persona], allowed
                )
                for persona in PERSONAS
            }
            if all(versions.values()):
                translated[locale] = versions
        projections: list[Projection] = []
        for persona in PERSONAS:
            original = written[persona]
            assert original is not None
            versions = [
                (locale, versions[persona]) for locale, versions in translated.items()
            ]
            texts, sources = compact(
                [
                    original["text"],
                    original["justification"],
                    *(
                        value
                        for _, version in versions
                        if version is not None
                        for value in (version["text"], version["justification"])
                    ),
                ],
                by_persona[persona].sources,
            )
            projections.append(
                Projection(
                    persona,
                    fit(texts[0] or "", PROJECTION_MAX_CHARS),
                    fit(texts[1] or "", PROJECTION_MAX_CHARS),
                    {
                        locale: {
                            "text": fit(
                                texts[2 + 2 * index] or "", PROJECTION_MAX_CHARS
                            ),
                            "justification": fit(
                                texts[3 + 2 * index] or "", PROJECTION_MAX_CHARS
                            ),
                        }
                        for index, (locale, _) in enumerate(versions)
                    },
                    sources,
                )
            )
        action_references = [
            *raw_action.excerpts,
            *(
                reference
                for persona in PERSONAS
                if (projection := raw_action.projections.get(persona)) is not None
                for reference in references(
                    f"{projection.text} {projection.justification}"
                )
            ),
        ]
        actions[raw_action.key] = Action(
            id=new_id(),
            finding=finding,
            segment_ids=_cited(batch, raw_action.key, action_references, spans)
            or finding.segment_ids,
            projections=projections,
        )
    restitution.actions = [actions[key] for key in batch.origins if key in actions]

    restitution.qualified = True
