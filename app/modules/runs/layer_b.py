"""Couche B : notes évaluatives 0-3 par critère (Algorithmes de calcul §5).

Déroulé, pour un run :

1. `retrieve` : les extraits de chaque sous-question sont les segments du
   corpus les plus proches (recherche vectorielle faite en SQL) au-dessus du
   seuil de similarité ;
2. `build_batches` : seuls les critères dont au moins 50 % des sous-questions
   ont des extraits sont soumis à l'IA (RG-1) ; les autres sont non conclus
   sans appel. Tout le run tient en un seul appel : extraits dédoublonnés,
   tronqués, désignés par des identifiants courts (E1, Q1, C1). Plusieurs
   appels seulement si le volume dépasse le plafond ;
3. `interpret` : une sous-question n'est documentée que si l'IA lui donne une
   note en citant au moins un de ses propres extraits (GF-1, RG-4) ;
4. `finalize` : note du critère = moyenne des notes des sous-questions
   documentées, si au moins 50 % le sont ; ambiguïté si les extraits se
   contredisent ou si les notes sont dispersées (RG-3).

Sans IA (désactivée, en erreur, plafond de coût), aucune note n'est
produite : le critère est non conclu (GF-3).

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

LAYER_B_VERSION = "layer-b/1.0.0"
PROMPT_CODE = "grade_sub_questions"
PROMPT_VERSION = "v1.0"

NoteStatus = Literal["conclu", "non_conclu"]
AnswerStatus = Literal["documentee", "non_documentee"]

# Tokens estimés : ~3 caractères par token pour du français en JSON compact
# (estimation prudente), plus le prompt système.
CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_TOKENS = 700
OUTPUT_TOKENS_PER_QUESTION = 30
OUTPUT_TOKENS_PER_CRITERION = 120
DEFINITION_CHARS = 200
EXPECTED_EVIDENCE_CHARS = 160
JUSTIFICATION_CHARS = 600


@dataclass(frozen=True, slots=True)
class LayerBRules:
    min_similarity: float
    top_k: int = 5
    excerpts_per_question: int = 3
    excerpt_chars: int = 600
    max_input_tokens: int = 30000
    max_output_tokens: int = 6000
    documented_ratio: float = 0.5
    ambiguity_std: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {"engine_version": LAYER_B_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class SubQuestion:
    id: UUID
    text: str
    expected_evidence: str | None = None


@dataclass(frozen=True, slots=True)
class Criterion:
    id: UUID
    code: str
    name: str
    definition: str | None
    applicability: str
    questions: tuple[SubQuestion, ...]


@dataclass(frozen=True, slots=True)
class LayerBInputs:
    locale: str
    cycle: str
    criteria: tuple[Criterion, ...]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "LayerBInputs":
        return cls(
            locale=row.get("locale") or "fr",
            cycle=row.get("cycle") or "",
            criteria=tuple(
                Criterion(
                    id=UUID(item["id"]),
                    code=item["code"],
                    name=item["name"],
                    definition=item.get("definition"),
                    applicability=item["applicability"],
                    questions=tuple(
                        SubQuestion(
                            UUID(question["id"]),
                            question["text"],
                            question.get("expected_evidence"),
                        )
                        for question in item.get("questions") or []
                    ),
                )
                for item in row.get("criteria") or []
            ),
        )

    def queries(self) -> list[tuple[UUID, str]]:
        """Texte vectorisé de chaque sous-question : le nom du critère situe
        la question, comme les sondes des piliers (« pilier : variable »)."""
        return [
            (question.id, f"{criterion.name} : {question.text}")
            for criterion in self.criteria
            for question in criterion.questions
        ]


@dataclass(frozen=True, slots=True)
class Candidate:
    """Segment retrouvé pour une sous-question."""

    sub_question_id: UUID
    segment_id: UUID
    document_id: UUID
    similarity: float
    text: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Candidate":
        return cls(
            UUID(row["sub_question_id"]),
            UUID(row["segment_id"]),
            UUID(row["document_id"]),
            float(row["similarity"]),
            row.get("text") or "",
        )


@dataclass(frozen=True, slots=True)
class Grade:
    note: int
    segment_ids: tuple[UUID, ...]
    contradictory: bool


@dataclass(frozen=True, slots=True)
class RawGrade:
    """Réponse brute de l'IA pour une sous-question (identifiants courts)."""

    key: str
    note: int | None
    excerpts: tuple[str, ...]
    contradictory: bool


@dataclass(frozen=True, slots=True)
class RawGrading:
    grades: tuple[RawGrade, ...]
    justifications: Mapping[str, str]


@dataclass(slots=True)
class GradingBatch:
    """Un appel IA : critères soumis, contenu du message et correspondance
    des identifiants courts."""

    criteria: list[Criterion] = field(default_factory=list)
    content: dict[str, Any] = field(default_factory=dict)
    criterion_keys: dict[str, UUID] = field(default_factory=dict)
    question_keys: dict[str, UUID] = field(default_factory=dict)
    excerpt_keys: dict[str, UUID] = field(default_factory=dict)
    # Extraits proposés pour chaque sous-question : seuls citables (GF-1).
    allowed: dict[UUID, frozenset[str]] = field(default_factory=dict)
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0

    def message(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class Grading:
    """Notes obtenues de l'IA, par sous-question et par critère."""

    graded_criteria: set[UUID] = field(default_factory=set)
    grades: dict[UUID, Grade] = field(default_factory=dict)
    justifications: dict[UUID, str] = field(default_factory=dict)


@dataclass(slots=True)
class SubAnswerResult:
    id: UUID
    sub_question_id: UUID
    status: AnswerStatus
    note: int | None
    contradictory: bool
    evidences: list[Candidate]


@dataclass(slots=True)
class CriterionNote:
    criterion_id: UUID
    note: float | None
    status: NoteStatus
    documented: int
    total: int
    justification: str
    ambiguous: bool


@dataclass(slots=True)
class LayerBResults:
    notes: list[CriterionNote]
    answers: list[SubAnswerResult]
    ai: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "criteria": len(self.notes),
            "concluded": sum(note.status == "conclu" for note in self.notes),
            "not_concluded": sum(note.status == "non_conclu" for note in self.notes),
            "sub_questions": len(self.answers),
            "documented": sum(answer.status == "documentee" for answer in self.answers),
            "ai_status": self.ai.get("status"),
            "ai_calls": self.ai.get("calls", 0),
        }

    def payload(self) -> dict[str, Any]:
        """Partie `layer_b` du corps de `complete_run`."""
        return {
            "notes": [
                {
                    "criterion_id": str(note.criterion_id),
                    "note": note.note,
                    "status": note.status,
                    "documented_sub_questions": note.documented,
                    "total_sub_questions": note.total,
                    "justification": note.justification,
                    "ambiguity_flag": note.ambiguous,
                }
                for note in self.notes
            ],
            "answers": [
                {
                    "id": str(answer.id),
                    "sub_question_id": str(answer.sub_question_id),
                    "status": answer.status,
                    "note": answer.note,
                    "contradictory": answer.contradictory,
                }
                for answer in self.answers
            ],
            "evidences": [
                {
                    "sub_answer_id": str(answer.id),
                    "segment_id": str(candidate.segment_id),
                    "similarity": candidate.similarity,
                }
                for answer in self.answers
                for candidate in answer.evidences
            ],
        }


def retrieve(
    candidates: Iterable[Candidate], rules: LayerBRules
) -> dict[UUID, list[Candidate]]:
    """Extraits retenus par sous-question : au-dessus du seuil, les plus
    proches d'abord, un segment une seule fois."""
    by_question: dict[UUID, dict[UUID, Candidate]] = defaultdict(dict)
    for candidate in candidates:
        if candidate.similarity < rules.min_similarity:
            continue
        kept = by_question[candidate.sub_question_id]
        current = kept.get(candidate.segment_id)
        if current is None or candidate.similarity > current.similarity:
            kept[candidate.segment_id] = candidate
    return {
        question_id: sorted(
            kept.values(),
            key=lambda candidate: (-candidate.similarity, str(candidate.segment_id)),
        )[: rules.top_k]
        for question_id, kept in by_question.items()
    }


def is_documented_enough(documented: int, total: int, ratio: float) -> bool:
    """RG-1 : au moins 50 % des sous-questions documentées."""
    return total > 0 and documented >= total * ratio


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut + "…"


def build_batches(
    inputs: LayerBInputs,
    retrieved: Mapping[UUID, list[Candidate]],
    rules: LayerBRules,
) -> list[GradingBatch]:
    """Lots d'appels IA : un seul en pratique. Un critère n'est jamais
    coupé entre deux appels ; un lot est fermé quand le critère suivant
    ferait dépasser le plafond d'entrée ou de sortie."""
    batches: list[GradingBatch] = []
    current = GradingBatch()
    for criterion in inputs.criteria:
        asked = [
            question for question in criterion.questions if retrieved.get(question.id)
        ]
        if not is_documented_enough(
            len(asked), len(criterion.questions), rules.documented_ratio
        ):
            continue
        candidate = _with_criterion(current, inputs, criterion, asked, retrieved, rules)
        if current.criteria and (
            candidate.estimated_input_tokens > rules.max_input_tokens
            or candidate.max_output_tokens > rules.max_output_tokens
        ):
            batches.append(current)
            candidate = _with_criterion(
                GradingBatch(), inputs, criterion, asked, retrieved, rules
            )
        current = candidate
    if current.criteria:
        batches.append(current)
    return batches


def _with_criterion(
    batch: GradingBatch,
    inputs: LayerBInputs,
    criterion: Criterion,
    questions: list[SubQuestion],
    retrieved: Mapping[UUID, list[Candidate]],
    rules: LayerBRules,
) -> GradingBatch:
    """Copie du lot augmentée d'un critère (le lot d'origine reste intact
    si le critère doit partir dans le lot suivant)."""
    excerpts: dict[str, str] = dict(batch.content.get("x", {}))
    excerpt_keys = dict(batch.excerpt_keys)
    key_of_segment = {segment: key for key, segment in excerpt_keys.items()}
    question_keys = dict(batch.question_keys)
    allowed = dict(batch.allowed)
    criterion_keys = dict(batch.criterion_keys)

    criterion_key = f"C{len(criterion_keys) + 1}"
    criterion_keys[criterion_key] = criterion.id
    asked: list[dict[str, Any]] = []
    for question in questions:
        question_key = f"Q{len(question_keys) + 1}"
        question_keys[question_key] = question.id
        keys: list[str] = []
        for candidate in retrieved[question.id][: rules.excerpts_per_question]:
            key = key_of_segment.get(candidate.segment_id)
            if key is None:
                key = f"E{len(excerpt_keys) + 1}"
                excerpt_keys[key] = candidate.segment_id
                key_of_segment[candidate.segment_id] = key
                excerpts[key] = truncate(candidate.text, rules.excerpt_chars)
            keys.append(key)
        allowed[question.id] = frozenset(keys)
        item: dict[str, Any] = {"k": question_key, "t": question.text, "e": keys}
        if question.expected_evidence:
            item["p"] = truncate(question.expected_evidence, EXPECTED_EVIDENCE_CHARS)
        asked.append(item)

    entry: dict[str, Any] = {
        "k": criterion_key,
        "n": criterion.name,
        "s": criterion.applicability,
        "q": asked,
    }
    if criterion.definition:
        entry["d"] = truncate(criterion.definition, DEFINITION_CHARS)
    content = {
        "lang": inputs.locale,
        "cycle": inputs.cycle,
        "x": excerpts,
        "c": [*batch.content.get("c", []), entry],
    }
    extended = GradingBatch(
        criteria=[*batch.criteria, criterion],
        content=content,
        criterion_keys=criterion_keys,
        question_keys=question_keys,
        excerpt_keys=excerpt_keys,
        allowed=allowed,
    )
    extended.estimated_input_tokens = SYSTEM_PROMPT_TOKENS + (
        len(extended.message()) // CHARS_PER_TOKEN
    )
    extended.max_output_tokens = OUTPUT_TOKENS_PER_QUESTION * len(
        question_keys
    ) + OUTPUT_TOKENS_PER_CRITERION * len(criterion_keys)
    return extended


def interpret(batch: GradingBatch, raw: RawGrading, grading: Grading) -> None:
    """Ajoute à `grading` les notes valides d'une réponse IA. Une note hors
    barème, sans extrait cité, ou citant un extrait qui n'était pas proposé
    pour la question, n'est pas retenue : la question reste non documentée."""
    grading.graded_criteria.update(criterion.id for criterion in batch.criteria)
    for raw_grade in raw.grades:
        question_id = batch.question_keys.get(raw_grade.key)
        if question_id is None or question_id in grading.grades:
            continue
        if raw_grade.note is None or raw_grade.note not in (0, 1, 2, 3):
            continue
        cited = [
            batch.excerpt_keys[key]
            for key in dict.fromkeys(raw_grade.excerpts)
            if key in batch.allowed.get(question_id, ())
        ]
        if not cited:
            continue
        grading.grades[question_id] = Grade(
            raw_grade.note, tuple(cited), raw_grade.contradictory
        )
    for key, text in raw.justifications.items():
        criterion_id = batch.criterion_keys.get(key)
        if criterion_id is not None and text and text.strip():
            grading.justifications[criterion_id] = truncate(text, JUSTIFICATION_CHARS)


JUSTIFICATIONS = {
    "fr": {
        "no_questions": "Aucune sous-question active.",
        "insufficient": (
            "Moins de 50 % des sous-questions sont documentées ({documented}/{total})."
        ),
        "ai_unavailable": (
            "Notation automatique indisponible : aucune note n'est produite "
            "({documented}/{total} sous-questions documentées)."
        ),
        "concluded": "Moyenne des notes de {documented} sous-questions documentées.",
    },
    "en": {
        "no_questions": "No active sub-question.",
        "insufficient": (
            "Fewer than 50% of the sub-questions are documented ({documented}/{total})."
        ),
        "ai_unavailable": (
            "Automatic rating unavailable: no rating is produced "
            "({documented}/{total} sub-questions documented)."
        ),
        "concluded": "Average rating of {documented} documented sub-questions.",
    },
}


def justification(locale: str, reason: str, documented: int, total: int) -> str:
    texts = JUSTIFICATIONS.get(locale, JUSTIFICATIONS["en"])
    return texts[reason].format(documented=documented, total=total)


def is_ambiguous(grades: list[Grade], threshold: float) -> bool:
    """RG-3 : extraits contradictoires, ou notes très dispersées."""
    if any(grade.contradictory for grade in grades):
        return True
    notes = [grade.note for grade in grades]
    return len(notes) >= 2 and statistics.pstdev(notes) > threshold


def finalize(
    inputs: LayerBInputs,
    retrieved: Mapping[UUID, list[Candidate]],
    grading: Grading,
    rules: LayerBRules,
    new_id: Callable[[], UUID] = uuid4,
) -> LayerBResults:
    notes: list[CriterionNote] = []
    answers: list[SubAnswerResult] = []
    for criterion in inputs.criteria:
        graded = criterion.id in grading.graded_criteria
        grades: list[Grade] = []
        documented = 0
        for question in criterion.questions:
            candidates = retrieved.get(question.id, [])
            grade = grading.grades.get(question.id) if graded else None
            if graded:
                # Notée par l'IA : documentée seulement si elle cite un extrait.
                is_documented = grade is not None
                cited = set(grade.segment_ids) if grade else set()
                evidences = [c for c in candidates if c.segment_id in cited]
            else:
                # Non soumise (critère sous 50 %, ou IA indisponible) : la
                # recherche seule dit si des extraits existent.
                is_documented = bool(candidates)
                evidences = list(candidates)
            if is_documented:
                documented += 1
            if grade is not None:
                grades.append(grade)
            answers.append(
                SubAnswerResult(
                    id=new_id(),
                    sub_question_id=question.id,
                    status="documentee" if is_documented else "non_documentee",
                    note=grade.note if grade else None,
                    contradictory=grade.contradictory if grade else False,
                    evidences=evidences,
                )
            )

        total = len(criterion.questions)
        if total == 0:
            reason = "no_questions"
        elif not is_documented_enough(documented, total, rules.documented_ratio):
            reason = "insufficient"
        elif not graded:
            reason = "ai_unavailable"
        else:
            reason = "concluded"
        if reason != "concluded":
            notes.append(
                CriterionNote(
                    criterion_id=criterion.id,
                    note=None,
                    status="non_conclu",
                    documented=documented,
                    total=total,
                    justification=justification(
                        inputs.locale, reason, documented, total
                    ),
                    ambiguous=False,
                )
            )
            continue
        notes.append(
            CriterionNote(
                criterion_id=criterion.id,
                note=round(statistics.fmean(grade.note for grade in grades), 1),
                status="conclu",
                documented=documented,
                total=total,
                justification=grading.justifications.get(criterion.id)
                or justification(inputs.locale, reason, documented, total),
                ambiguous=is_ambiguous(grades, rules.ambiguity_std),
            )
        )
    return LayerBResults(notes=notes, answers=answers)
