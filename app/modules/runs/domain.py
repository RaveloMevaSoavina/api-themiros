"""Couche A : variables intermédiaires, confiance corpus et scores objectivés
(Algorithmes de calcul §3, §4, §7).

Fonctions pures : aucune dépendance au stockage ni au réseau.

Notation d'une variable (règle déterministe retenue le 20/10/2026, la spec ne
définissant pas `isExplicitAndStructured` / `isWeakMention`). Une détection
est un segment classé dans le pilier de la variable dont la similarité avec la
sonde de la variable dépasse le seuil de classification
(`segment_variable_matches`). Pour une version de programme :

- 1 (`present`) : détections fortes (similarité ≥ `strong_similarity`) dans
  au moins `min_independent_sources` documents distincts ;
- 0,5 (`partiel`) : au moins une détection, sans cette convergence ;
- vide (`non_renseigne`) : aucune détection.

Le moteur ne conclut jamais à l'absence (0) : une preuve textuelle montre ce
qui est présent, pas ce qui manque (GF-3).
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

ENGINE_VERSION = "layer-a/1.0.0"

VariableState = Literal["present", "partiel", "absent", "non_renseigne"]
EvolutionLabel = Literal["renforce", "maintenu", "affaibli"]
ClassificationState = Literal["ready", "pending", "failed", "framework_changed"]


@dataclass(frozen=True, slots=True)
class ScoringRules:
    strong_similarity: float
    min_independent_sources: int
    # P(v | document) = min(1, segments convergents / 3) (§7.4).
    convergence_segments: int = 3
    evolution_threshold: float = 5
    max_evidences_per_variable: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {"engine_version": ENGINE_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class Variable:
    id: UUID
    code: str
    weight: float


@dataclass(frozen=True, slots=True)
class Pillar:
    id: UUID
    variables: tuple[Variable, ...]


@dataclass(frozen=True, slots=True)
class ProgramVersion:
    id: UUID
    order_index: int


@dataclass(frozen=True, slots=True)
class Match:
    segment_id: UUID
    document_id: UUID
    pillar_variable_id: UUID
    similarity: float


@dataclass(frozen=True, slots=True)
class RunInputs:
    state: ClassificationState
    created_at: datetime
    versions: tuple[ProgramVersion, ...]
    # Versions documentées par chaque document du corpus figé.
    document_versions: dict[UUID, frozenset[UUID]]
    unassigned_documents: int
    pillars: tuple[Pillar, ...]
    segments: dict[str, int]
    matches: tuple[Match, ...]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RunInputs":
        document_versions: dict[UUID, set[UUID]] = defaultdict(set)
        unassigned: set[UUID] = set()
        for item in row.get("documents") or []:
            document_id = UUID(item["document_id"])
            if item.get("program_version_id"):
                document_versions[document_id].add(UUID(item["program_version_id"]))
            else:
                unassigned.add(document_id)
        return cls(
            state=row["state"],
            created_at=datetime.fromisoformat(row["created_at"]),
            versions=tuple(
                ProgramVersion(UUID(item["id"]), int(item["order_index"]))
                for item in row.get("versions") or []
            ),
            document_versions={
                document_id: frozenset(versions)
                for document_id, versions in document_versions.items()
            },
            unassigned_documents=len(unassigned - document_versions.keys()),
            pillars=tuple(
                Pillar(
                    UUID(item["id"]),
                    tuple(
                        Variable(
                            UUID(variable["id"]),
                            variable["code"],
                            float(variable["weight"]),
                        )
                        for variable in item.get("variables") or []
                    ),
                )
                for item in row.get("pillars") or []
            ),
            segments={
                key: int(value) for key, value in (row.get("segments") or {}).items()
            },
            matches=tuple(
                Match(
                    UUID(item["segment_id"]),
                    UUID(item["document_id"]),
                    UUID(item["pillar_variable_id"]),
                    float(item["similarity"]),
                )
                for item in row.get("matches") or []
            ),
        )


@dataclass(slots=True)
class VariableResult:
    id: UUID
    pillar_id: UUID
    variable_code: str
    program_version_id: UUID
    value: float | None
    state: VariableState
    documents_count: int
    corpus_confidence: float | None
    evidences: list[Match] = field(default_factory=list)


@dataclass(slots=True)
class PillarScore:
    pillar_id: UUID
    program_version_id: UUID
    score: float | None
    corpus_confidence: float
    documents_count: int
    delta: float | None = None
    evolution_label: EvolutionLabel | None = None


@dataclass(frozen=True, slots=True)
class LayerAResults:
    variables: list[VariableResult]
    scores: list[PillarScore]

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            "variables": {
                "variables_computed": len(self.variables),
                "variables_non_renseigne": sum(
                    variable.state == "non_renseigne" for variable in self.variables
                ),
                "evidences": sum(
                    len(variable.evidences) for variable in self.variables
                ),
            },
            "layer_a": {
                "scores": len(self.scores),
                "scores_without_value": sum(
                    score.score is None for score in self.scores
                ),
            },
        }

    def payload(self, rules: ScoringRules) -> dict[str, Any]:
        """Corps de `complete_run`."""
        return {
            "variables": [
                {
                    "id": str(variable.id),
                    "pillar_id": str(variable.pillar_id),
                    "variable_code": variable.variable_code,
                    "program_version_id": str(variable.program_version_id),
                    "value": variable.value,
                    "state": variable.state,
                    "documents_count": variable.documents_count,
                    "corpus_confidence": variable.corpus_confidence,
                }
                for variable in self.variables
            ],
            "evidences": [
                {
                    "variable_id": str(variable.id),
                    "segment_id": str(match.segment_id),
                    "similarity": match.similarity,
                }
                for variable in self.variables
                for match in variable.evidences
            ],
            "scores": [
                {
                    "pillar_id": str(score.pillar_id),
                    "program_version_id": str(score.program_version_id),
                    "score": score.score,
                    "delta": score.delta,
                    "evolution_label": score.evolution_label,
                    "corpus_confidence": score.corpus_confidence,
                    "documents_count": score.documents_count,
                }
                for score in self.scores
            ],
            "rules": rules.as_dict(),
        }


def variable_value(detections: Iterable[Match], rules: ScoringRules) -> float | None:
    """1, 0,5 ou `None` (non renseignée) ; jamais 0 (voir l'en-tête)."""
    detections = list(detections)
    if not detections:
        return None
    strong_documents = {
        match.document_id
        for match in detections
        if match.similarity >= rules.strong_similarity
    }
    return 1.0 if len(strong_documents) >= rules.min_independent_sources else 0.5


def variable_state(value: float | None) -> VariableState:
    if value is None:
        return "non_renseigne"
    if value >= 1:
        return "present"
    if value >= 0.5:
        return "partiel"
    return "absent"


def corpus_confidence(
    detections: Iterable[Match], total_documents: int, convergence_segments: int = 3
) -> float | None:
    """P(v | corpus) = 1 − Π (1 − P(v | document)), avec P(v | document) =
    min(1, segments convergents / 3). Un document sans détection contribue
    P = 0, donc un facteur 1 (§7.4)."""
    if total_documents == 0:
        return None
    segments_by_document: dict[UUID, set[UUID]] = defaultdict(set)
    for match in detections:
        segments_by_document[match.document_id].add(match.segment_id)
    product = 1.0
    for segments in segments_by_document.values():
        product *= 1.0 - min(1.0, len(segments) / convergence_segments)
    return round(1.0 - product, 4)


def pillar_score(variables: Iterable[tuple[float, float | None]]) -> float | None:
    """Σ (poids × valeur) / Σ poids × 100, sur les variables renseignées ;
    `None` sans variable renseignée (RG-4, RG-5)."""
    total_weight = 0.0
    weighted_sum = 0.0
    for weight, value in variables:
        if value is None:
            continue
        total_weight += weight
        weighted_sum += weight * value
    if total_weight <= 0:
        return None
    return round(weighted_sum / total_weight * 100, 2)


def pillar_confidence(confidences: Iterable[float | None]) -> float:
    """Moyenne des confiances des variables ; une variable non renseignée
    compte pour 0 et réduit la confiance du pilier (§4.7, RG-5)."""
    values = [confidence or 0.0 for confidence in confidences]
    return round(sum(values) / len(values), 4) if values else 0.0


def evolution_label(delta: float, threshold: float = 5) -> EvolutionLabel:
    if delta >= threshold:
        return "renforce"
    if delta <= -threshold:
        return "affaibli"
    return "maintenu"


def select_evidences(detections: Iterable[Match], limit: int) -> list[Match]:
    """Meilleur extrait de chaque document d'abord, puis les suivants par
    similarité : la preuve montre la convergence des sources."""
    ordered = sorted(
        detections,
        key=lambda match: (
            -match.similarity,
            str(match.document_id),
            str(match.segment_id),
        ),
    )
    seen: set[UUID] = set()
    first_per_document: list[Match] = []
    others: list[Match] = []
    for match in ordered:
        if match.document_id in seen:
            others.append(match)
        else:
            seen.add(match.document_id)
            first_per_document.append(match)
    return (first_per_document + others)[:limit]


def compute_layer_a(
    inputs: RunInputs,
    rules: ScoringRules,
    new_id: Callable[[], UUID] = uuid4,
) -> LayerAResults:
    versions = sorted(inputs.versions, key=lambda version: version.order_index)
    documents_per_version: dict[UUID, int] = defaultdict(int)
    for document_versions in inputs.document_versions.values():
        for version_id in document_versions:
            documents_per_version[version_id] += 1
    matches_by_variable: dict[UUID, list[Match]] = defaultdict(list)
    for match in inputs.matches:
        matches_by_variable[match.pillar_variable_id].append(match)

    variables: list[VariableResult] = []
    scores: list[PillarScore] = []
    for pillar in inputs.pillars:
        pillar_scores: list[PillarScore] = []
        for version in versions:
            results: list[tuple[Variable, VariableResult]] = []
            pillar_documents: set[UUID] = set()
            for variable in pillar.variables:
                detections = [
                    match
                    for match in matches_by_variable[variable.id]
                    if version.id in inputs.document_versions.get(match.document_id, ())
                ]
                value = variable_value(detections, rules)
                documents = {match.document_id for match in detections}
                pillar_documents |= documents
                results.append(
                    (
                        variable,
                        VariableResult(
                            id=new_id(),
                            pillar_id=pillar.id,
                            variable_code=variable.code,
                            program_version_id=version.id,
                            value=value,
                            state=variable_state(value),
                            documents_count=len(documents),
                            corpus_confidence=(
                                corpus_confidence(
                                    detections,
                                    documents_per_version[version.id],
                                    rules.convergence_segments,
                                )
                                if detections
                                else None
                            ),
                            evidences=select_evidences(
                                detections, rules.max_evidences_per_variable
                            ),
                        ),
                    )
                )
            variables.extend(result for _, result in results)
            pillar_scores.append(
                PillarScore(
                    pillar_id=pillar.id,
                    program_version_id=version.id,
                    score=pillar_score(
                        (variable.weight, result.value) for variable, result in results
                    ),
                    corpus_confidence=pillar_confidence(
                        result.corpus_confidence for _, result in results
                    ),
                    documents_count=len(pillar_documents),
                )
            )
        # Évolution par rapport à la version précédente (§4.4, étape 6).
        for previous, current in zip(pillar_scores, pillar_scores[1:], strict=False):
            if previous.score is not None and current.score is not None:
                current.delta = round(current.score - previous.score, 2)
                current.evolution_label = evolution_label(
                    current.delta, rules.evolution_threshold
                )
        scores.extend(pillar_scores)
    return LayerAResults(variables=variables, scores=scores)
