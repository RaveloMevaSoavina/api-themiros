"""Couche C : alertes de cohérence entre couches A et B (Algorithmes de
calcul §6, Spécification des prompts IA §6).

Déroulé, pour un run :

1. `detect` : règles déterministes sur les résultats déjà calculés du run.
   Pour chaque lien pilier ↔ critère retenu, le score A de la dernière
   version scorée du pilier est comparé à la note B du critère :

   - règle propre au critère d'abord (ALERT-03 efficience, ALERT-04 équité,
     ALERT-05 gestion adaptative : A au-dessus de son seuil, B ≤ 1) ;
   - puis ALERT-01 (A ≥ 60, B ≤ 1) et ALERT-02 (A ≤ 40, B ≥ 2).

   Un critère non conclu ne produit aucune alerte : un corpus pauvre réduit
   la certitude, il ne fabrique pas d'écart (RG-2). ALERT-06 signale un
   pilier « renforcé » d'une version à l'autre avec une confiance corpus
   sous 0,5. Toute alerte sur un corpus pauvre est mineure ; une alerte
   sans preuve n'est pas créée (RG-4). Aucun score n'est modifié (RG-3) ;
2. `build_batches` : tous les écarts du run en un seul appel IA (prompt
   detect_incoherence), extraits dédoublonnés et tronqués, identifiants
   courts (A1, E1). Plusieurs appels seulement au-delà du plafond ;
3. `interpret` : l'IA confirme ou infirme chaque écart, fixe sa gravité et
   rédige son message en citant des extraits proposés pour lui. Un écart
   infirmé n'est pas publié.

Sans IA (désactivée, en erreur, plafond de coût), les écarts détectés sont
publiés avec leur gravité par défaut et le libellé de leur règle, marqués
non qualifiés : une alerte avertit sans rien noter.

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from app.modules.runs.domain import LayerAResults, PillarScore
from app.modules.runs.layer_b import LayerBResults, truncate

LAYER_C_VERSION = "layer-c/1.0.0"
PROMPT_CODE = "detect_incoherence"
PROMPT_VERSION = "v1.0"

Severity = Literal["mineure", "majeure"]

# Mêmes estimations que la couche B (~3 caractères par token).
CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_TOKENS = 600
OUTPUT_TOKENS_PER_ALERT = 90
MESSAGE_CHARS = 200
STORED_MESSAGE_CHARS = 300


@dataclass(frozen=True, slots=True)
class LayerCRules:
    high_score: float = 60
    low_score: float = 40
    high_note: float = 2
    low_note: float = 1
    efficiency_score: float = 50
    equity_score: float = 40
    monitoring_score: float = 50
    poor_confidence: float = 0.5
    evidences_per_alert: int = 4
    excerpt_chars: int = 500
    max_input_tokens: int = 30000
    max_output_tokens: int = 2000

    def as_dict(self) -> dict[str, Any]:
        return {"engine_version": LAYER_C_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class Rule:
    code: str
    alert_type: str
    severity: Severity


RULES = {
    rule.code: rule
    for rule in (
        Rule("ALERT-01", "documentaire_sans_credibilite", "majeure"),
        Rule("ALERT-02", "jugement_a_verifier", "majeure"),
        Rule("ALERT-03", "financement_non_operationnalise", "mineure"),
        Rule("ALERT-04", "equite_discursive", "mineure"),
        Rule("ALERT-05", "suivi_insuffisamment_instrumente", "mineure"),
        Rule("ALERT-06", "progres_documentaire_sans_confiance", "mineure"),
    )
}

# Règles propres à un critère : code du critère -> (règle, seuil du score A).
CRITERION_RULES = {
    "efficience": ("ALERT-03", "efficiency_score"),
    "equite": ("ALERT-04", "equity_score"),
    "gestion_adaptative": ("ALERT-05", "monitoring_score"),
}


@dataclass(frozen=True, slots=True)
class PillarCriterion:
    pillar_id: UUID
    criterion_id: UUID
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class LayerCInputs:
    locale: str
    pillar_names: Mapping[UUID, str]
    links: tuple[PillarCriterion, ...]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "LayerCInputs":
        return cls(
            locale=row.get("locale") or "fr",
            pillar_names={
                UUID(item["id"]): item["name"] for item in row.get("pillars") or []
            },
            links=tuple(
                PillarCriterion(
                    UUID(item["pillar_id"]),
                    UUID(item["criterion_id"]),
                    item["code"],
                    item["name"],
                )
                for item in row.get("links") or []
            ),
        )


@dataclass(slots=True)
class Alert:
    id: UUID
    rule: Rule
    severity: Severity
    pillar_id: UUID
    pillar_name: str
    criterion_id: UUID | None
    criterion_name: str | None
    program_version_id: UUID
    score: float
    note: float | None
    delta: float | None
    confidence: float
    segment_ids: list[UUID]
    message: str
    qualified: bool = False


@dataclass(frozen=True, slots=True)
class RawQualification:
    """Réponse brute de l'IA pour un écart (identifiants courts)."""

    key: str
    confirmed: bool
    severity: str
    message: str
    excerpts: tuple[str, ...]


@dataclass(slots=True)
class QualificationBatch:
    """Un appel IA : écarts soumis, contenu du message et correspondance
    des identifiants courts."""

    alerts: list[Alert] = field(default_factory=list)
    content: dict[str, Any] = field(default_factory=dict)
    alert_keys: dict[str, UUID] = field(default_factory=dict)
    excerpt_keys: dict[str, UUID] = field(default_factory=dict)
    # Extraits proposés pour chaque écart : seuls citables (GF-1).
    allowed: dict[UUID, frozenset[str]] = field(default_factory=dict)
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0

    def message(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class LayerCResults:
    alerts: list[Alert]
    detected: int
    ai: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "alerts": len(self.alerts),
            "major": sum(alert.severity == "majeure" for alert in self.alerts),
            "infirmed": self.detected - len(self.alerts),
            "qualified": sum(alert.qualified for alert in self.alerts),
            "ai_status": self.ai.get("status"),
            "ai_calls": self.ai.get("calls", 0),
        }

    def payload(self) -> dict[str, Any]:
        """Partie `layer_c` du corps de `complete_run`."""
        return {
            "alerts": [
                {
                    "id": str(alert.id),
                    "pillar_id": str(alert.pillar_id),
                    "criterion_id": (
                        str(alert.criterion_id) if alert.criterion_id else None
                    ),
                    "program_version_id": str(alert.program_version_id),
                    "rule_code": alert.rule.code,
                    "alert_type": alert.rule.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "qualified": alert.qualified,
                    "segment_ids": [str(segment) for segment in alert.segment_ids],
                }
                for alert in self.alerts
            ]
        }


MESSAGES = {
    "fr": {
        "ALERT-01": (
            "{pillar} : amélioration documentaire sans crédibilité opérationnelle "
            "suffisante (score A {score}/100, {criterion} {note}/3)."
        ),
        "ALERT-02": (
            "{pillar} : jugement évaluatif à vérifier ({criterion} {note}/3 pour "
            "un score A de {score}/100)."
        ),
        "ALERT-03": (
            "{pillar} : financement non opérationnalisé (score A {score}/100, "
            "{criterion} {note}/3)."
        ),
        "ALERT-04": (
            "{pillar} : équité discursive (score A {score}/100, {criterion} {note}/3)."
        ),
        "ALERT-05": (
            "{pillar} : suivi insuffisamment instrumenté (score A {score}/100, "
            "{criterion} {note}/3)."
        ),
        "ALERT-06": (
            "{pillar} : progrès documentaire sans confiance corpus suffisante "
            "(+{delta} points, confiance {confidence} %)."
        ),
    },
    "en": {
        "ALERT-01": (
            "{pillar}: documentary improvement without sufficient operational "
            "credibility (layer A score {score}/100, {criterion} {note}/3)."
        ),
        "ALERT-02": (
            "{pillar}: evaluative judgement to verify ({criterion} {note}/3 for a "
            "layer A score of {score}/100)."
        ),
        "ALERT-03": (
            "{pillar}: funding not operationalised (layer A score {score}/100, "
            "{criterion} {note}/3)."
        ),
        "ALERT-04": (
            "{pillar}: equity in discourse only (layer A score {score}/100, "
            "{criterion} {note}/3)."
        ),
        "ALERT-05": (
            "{pillar}: monitoring insufficiently instrumented (layer A score "
            "{score}/100, {criterion} {note}/3)."
        ),
        "ALERT-06": (
            "{pillar}: documentary progress without sufficient corpus confidence "
            "(+{delta} points, confidence {confidence}%)."
        ),
    },
}


def default_message(locale: str, alert: Alert) -> str:
    texts = MESSAGES.get(locale, MESSAGES["en"])
    return truncate(
        texts[alert.rule.code].format(
            pillar=alert.pillar_name,
            criterion=alert.criterion_name or "",
            score=f"{alert.score:.0f}",
            note=f"{alert.note:.1f}" if alert.note is not None else "",
            delta=f"{alert.delta:.0f}" if alert.delta is not None else "",
            confidence=f"{alert.confidence * 100:.0f}",
        ),
        STORED_MESSAGE_CHARS,
    )


def match_rule(
    score: float, note: float, criterion_code: str, rules: LayerCRules
) -> Rule | None:
    """Règle propre au critère d'abord : son libellé dit mieux l'écart."""
    specific = CRITERION_RULES.get(criterion_code)
    if specific is not None:
        code, threshold = specific
        if score >= getattr(rules, threshold) and note <= rules.low_note:
            return RULES[code]
    if score >= rules.high_score and note <= rules.low_note:
        return RULES["ALERT-01"]
    if score <= rules.low_score and note >= rules.high_note:
        return RULES["ALERT-02"]
    return None


def ordered_segments(items: Iterable[tuple[UUID, float]]) -> list[UUID]:
    """Segments les plus proches d'abord, chacun une seule fois."""
    best: dict[UUID, float] = {}
    for segment_id, similarity in items:
        if similarity > best.get(segment_id, -1.0):
            best[segment_id] = similarity
    return sorted(best, key=lambda segment: (-best[segment], str(segment)))


def detect(
    inputs: LayerCInputs,
    layer_a: LayerAResults,
    layer_b: LayerBResults | None,
    rules: LayerCRules,
    new_id: Callable[[], UUID] = uuid4,
) -> list[Alert]:
    # Preuves de la couche A par pilier et version, de la couche B par critère.
    pillar_matches: dict[tuple[UUID, UUID], list[tuple[UUID, float]]] = defaultdict(
        list
    )
    for variable in layer_a.variables:
        pillar_matches[(variable.pillar_id, variable.program_version_id)].extend(
            (match.segment_id, match.similarity) for match in variable.evidences
        )
    criterion_matches: dict[UUID, list[tuple[UUID, float]]] = defaultdict(list)
    notes: dict[UUID, float] = {}
    if layer_b is not None:
        for answer in layer_b.answers:
            if answer.status == "documentee" and answer.criterion_id is not None:
                criterion_matches[answer.criterion_id].extend(
                    (candidate.segment_id, candidate.similarity)
                    for candidate in answer.evidences
                )
        notes = {
            note.criterion_id: note.note
            for note in layer_b.notes
            if note.status == "conclu" and note.note is not None
        }
    links_by_pillar: dict[UUID, list[PillarCriterion]] = defaultdict(list)
    for link in inputs.links:
        links_by_pillar[link.pillar_id].append(link)
    # Versions dans l'ordre (compute_layer_a les trie par order_index).
    scores_by_pillar: dict[UUID, list[PillarScore]] = defaultdict(list)
    for score in layer_a.scores:
        scores_by_pillar[score.pillar_id].append(score)

    def evidences(pillar: PillarScore, criterion_id: UUID | None = None) -> list[UUID]:
        segments = ordered_segments(criterion_matches.get(criterion_id, ()))
        segments += [
            segment
            for segment in ordered_segments(
                pillar_matches.get((pillar.pillar_id, pillar.program_version_id), ())
            )
            if segment not in segments
        ]
        return segments[: rules.evidences_per_alert]

    def alert(
        rule: Rule,
        pillar: PillarScore,
        link: PillarCriterion | None = None,
        note: float | None = None,
    ) -> Alert:
        assert pillar.score is not None
        poor = pillar.corpus_confidence < rules.poor_confidence
        created = Alert(
            id=new_id(),
            rule=rule,
            severity="mineure" if poor else rule.severity,
            pillar_id=pillar.pillar_id,
            pillar_name=inputs.pillar_names.get(pillar.pillar_id, ""),
            criterion_id=link.criterion_id if link else None,
            criterion_name=link.name if link else None,
            program_version_id=pillar.program_version_id,
            score=pillar.score,
            note=note,
            delta=pillar.delta,
            confidence=pillar.corpus_confidence,
            segment_ids=evidences(pillar, link.criterion_id if link else None),
            message="",
        )
        created.message = default_message(inputs.locale, created)
        return created

    alerts: list[Alert] = []
    for pillar_id, scores in scores_by_pillar.items():
        scored = [score for score in scores if score.score is not None]
        if scored:
            latest = scored[-1]
            assert latest.score is not None
            for link in links_by_pillar.get(pillar_id, ()):
                note = notes.get(link.criterion_id)
                if note is None:
                    continue
                rule = match_rule(latest.score, note, link.code, rules)
                if rule is not None:
                    alerts.append(alert(rule, latest, link, note))
        for score in scored:
            if (
                score.evolution_label == "renforce"
                and score.corpus_confidence < rules.poor_confidence
            ):
                alerts.append(alert(RULES["ALERT-06"], score))
    return [item for item in alerts if item.segment_ids]


def known_texts(layer_b: LayerBResults | None) -> dict[UUID, str]:
    """Texte des extraits déjà lus par la couche B."""
    if layer_b is None:
        return {}
    return {
        candidate.segment_id: candidate.text
        for answer in layer_b.answers
        for candidate in answer.evidences
        if candidate.text
    }


def _entry(
    batch: QualificationBatch,
    key: str,
    alert: Alert,
    texts: Mapping[UUID, str],
    rules: LayerCRules,
) -> tuple[dict[str, Any], dict[str, tuple[UUID, str]]]:
    """Entrée d'un écart et extraits nouveaux pour le lot."""
    key_of_segment = {segment: k for k, segment in batch.excerpt_keys.items()}
    added: dict[str, tuple[UUID, str]] = {}
    keys: list[str] = []
    for segment_id in alert.segment_ids:
        if segment_id not in texts:
            continue
        excerpt_key = key_of_segment.get(segment_id)
        if excerpt_key is None:
            excerpt_key = f"E{len(batch.excerpt_keys) + len(added) + 1}"
            key_of_segment[segment_id] = excerpt_key
            added[excerpt_key] = (
                segment_id,
                truncate(texts[segment_id], rules.excerpt_chars),
            )
        keys.append(excerpt_key)
    entry: dict[str, Any] = {
        "k": key,
        "r": alert.rule.code,
        "t": alert.rule.alert_type,
        "p": alert.pillar_name,
        "s": round(alert.score, 1),
        "cf": round(alert.confidence, 2),
        "g": alert.severity,
        "e": keys,
    }
    if alert.criterion_name:
        entry["c"] = alert.criterion_name
    if alert.note is not None:
        entry["n"] = round(alert.note, 1)
    if alert.rule.code == "ALERT-06" and alert.delta is not None:
        entry["d"] = round(alert.delta, 1)
    return entry, added


def _add(
    batch: QualificationBatch,
    alert: Alert,
    entry: dict[str, Any],
    added: Mapping[str, tuple[UUID, str]],
) -> None:
    batch.alerts.append(alert)
    batch.alert_keys[entry["k"]] = alert.id
    excerpts = batch.content.setdefault("x", {})
    for key, (segment_id, text) in added.items():
        batch.excerpt_keys[key] = segment_id
        excerpts[key] = text
    batch.allowed[alert.id] = frozenset(entry["e"])
    batch.content.setdefault("a", []).append(entry)
    batch.estimated_input_tokens = SYSTEM_PROMPT_TOKENS + (
        len(batch.message()) // CHARS_PER_TOKEN
    )
    batch.max_output_tokens = OUTPUT_TOKENS_PER_ALERT * len(batch.alerts)


def build_batches(
    inputs: LayerCInputs,
    alerts: Iterable[Alert],
    texts: Mapping[UUID, str],
    rules: LayerCRules,
) -> list[QualificationBatch]:
    """Lots d'appels IA : un seul en pratique. Un lot est fermé quand
    l'écart suivant ferait dépasser le plafond d'entrée ou de sortie."""
    batches: list[QualificationBatch] = []
    current = QualificationBatch(content={"lang": inputs.locale})
    for alert in alerts:
        key = f"A{len(current.alerts) + 1}"
        entry, added = _entry(current, key, alert, texts, rules)
        growth = len(
            json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        ) + sum(len(text) + len(key) + 6 for key, (_, text) in added.items())
        if current.alerts and (
            current.estimated_input_tokens + growth // CHARS_PER_TOKEN
            > rules.max_input_tokens
            or current.max_output_tokens + OUTPUT_TOKENS_PER_ALERT
            > rules.max_output_tokens
        ):
            batches.append(current)
            current = QualificationBatch(content={"lang": inputs.locale})
            entry, added = _entry(current, "A1", alert, texts, rules)
        _add(current, alert, entry, added)
    if current.alerts:
        batches.append(current)
    return batches


def interpret(
    batch: QualificationBatch,
    raw: Iterable[RawQualification],
    rules: LayerCRules,
) -> set[UUID]:
    """Applique une réponse IA aux écarts du lot ; renvoie les écarts
    infirmés. Gravité hors liste : celle par défaut. Corpus pauvre : mineure,
    quoi que dise l'IA (RG-2). Extraits cités hors de ceux proposés pour
    l'écart : ignorés ; sans extrait valable, l'écart garde ses preuves."""
    by_id = {alert.id: alert for alert in batch.alerts}
    seen: set[UUID] = set()
    infirmed: set[UUID] = set()
    for item in raw:
        alert_id = batch.alert_keys.get(item.key)
        if alert_id is None or alert_id in seen:
            continue
        seen.add(alert_id)
        if not item.confirmed:
            infirmed.add(alert_id)
            continue
        alert = by_id[alert_id]
        if item.severity in ("mineure", "majeure"):
            alert.severity = item.severity  # type: ignore[assignment]
        if alert.confidence < rules.poor_confidence:
            alert.severity = "mineure"
        if item.message and item.message.strip():
            alert.message = truncate(item.message, MESSAGE_CHARS)
        cited = [
            batch.excerpt_keys[key]
            for key in dict.fromkeys(item.excerpts)
            if key in batch.allowed.get(alert_id, ())
        ]
        if cited:
            alert.segment_ids = cited
        alert.qualified = True
    return infirmed
