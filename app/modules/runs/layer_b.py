"""Couche B : notes évaluatives 0-3 par critère (Algorithmes de calcul §5).

Déroulé, pour un run :

1. `retrieve` : les extraits de chaque sous-question sont les segments du
   corpus les plus proches (recherche vectorielle faite en SQL) au-dessus du
   seuil de similarité ;
2. `build_batches` : seuls les critères dont au moins 50 % des sous-questions
   ont des extraits sont soumis à l'IA (RG-1) ; les autres sont non conclus
   sans appel. Tout le run tient en un seul appel : extraits dédoublonnés,
   découpés en phrases, désignés par des identifiants courts (E1, Q1, C1).
   Plusieurs appels seulement si le volume dépasse le plafond ;
3. `interpret` : une sous-question n'est documentée que si l'IA lui donne une
   note en citant au moins un de ses propres extraits (GF-1, RG-4). L'IA
   cite des phrases (E1.2) : la preuve garde leurs positions, que le
   surlignage reprend. Chaque sous-question reçoit un paragraphe qui
   interprète sa réponse, chaque critère une justification argumentée ;
   tous deux citent leurs sources dans le texte et sont traduits dans les
   autres langues de l'interface par le même appel (`tr`). Un texte trop
   court ou sans source n'est pas publié. Les extraits eux-mêmes sont lus
   par l'étape de lecture (`reading`), une fois par run. Le projet évalué
   (`pj`) accompagne le message : les textes parlent de lui, pas des
   critères ni des notes ;
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

from app.modules.evidences.passages import Span, merge
from app.modules.frameworks.schemas import INTERFACE_LOCALES
from app.modules.runs.citations import (
    Excerpt,
    Reading,
    cited,
    excerpt,
    reading_payload,
    written,
)
from app.modules.runs.project import ProjectContext

LAYER_B_VERSION = "layer-b/1.3.0"
PROMPT_CODE = "grade_sub_questions"
# v1.1 : extraits en phrases, phrases citées, interprétation de chaque
# extrait cité, sources dans la justification. v1.2 : le projet évalué
# (`pj`), sujet des justifications et des interprétations, qui citent
# leurs phrases. v1.3 : interprétation en paragraphe de chaque
# sous-question, justification argumentée, traductions (`tr`) ; les
# extraits sont lus par l'étape de lecture.
PROMPT_VERSION = "v1.3"

NoteStatus = Literal["conclu", "non_conclu"]
AnswerStatus = Literal["documentee", "non_documentee"]

# Tokens estimés : ~3 caractères par token pour du français en JSON compact
# (estimation prudente), plus le prompt système.
CHARS_PER_TOKEN = 3
SYSTEM_PROMPT_TOKENS = 1500
# Note, extraits cités et interprétation (un paragraphe), par langue.
OUTPUT_TOKENS_PER_QUESTION = 260
# Justification (un paragraphe argumenté), par langue.
OUTPUT_TOKENS_PER_CRITERION = 480
DEFINITION_CHARS = 200
EXPECTED_EVIDENCE_CHARS = 160
# Le prompt (v1.3) demande 500 à 1 200 caractères hors sources pour une
# justification, 300 à 600 pour l'interprétation d'une sous-question : ces
# plafonds ne coupent qu'un dépassement net ; sous les minimums, une phrase
# de circonstance n'est pas publiée.
JUSTIFICATION_CHARS = 1400
JUSTIFICATION_MIN_CHARS = 200
INTERPRETATION_CHARS = 800
INTERPRETATION_MIN_CHARS = 120
# Plafonds de la base, marqueurs compris.
JUSTIFICATION_MAX_CHARS = 2000
INTERPRETATION_MAX_CHARS = 1500


@dataclass(frozen=True, slots=True)
class LayerBRules:
    min_similarity: float
    top_k: int = 5
    excerpts_per_question: int = 3
    excerpt_chars: int = 600
    max_input_tokens: int = 30000
    max_output_tokens: int = 16000
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
    project: ProjectContext = field(default_factory=ProjectContext)

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
    # Par segment cité : positions des phrases citées.
    spans: Mapping[UUID, tuple[Span, ...]] = field(default_factory=dict)
    # Interprétation de la réponse, sourcée et traduite.
    interpretation: Reading | None = None


@dataclass(frozen=True, slots=True)
class RawGrade:
    """Réponse brute de l'IA pour une sous-question (identifiants courts)."""

    key: str
    note: int | None
    excerpts: tuple[str, ...]
    contradictory: bool
    # Interprétation de la réponse (un paragraphe).
    interpretation: str = ""


@dataclass(frozen=True, slots=True)
class RawGradingTranslation:
    """Traduction de la réponse dans une langue, par identifiant court."""

    interpretations: Mapping[str, str] = field(default_factory=dict)
    justifications: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawGrading:
    grades: tuple[RawGrade, ...]
    justifications: Mapping[str, str]
    # Langue -> traduction ; seules les langues demandées (`tr`) comptent.
    translations: Mapping[str, RawGradingTranslation] = field(default_factory=dict)


@dataclass(slots=True)
class GradingBatch:
    """Un appel IA : critères soumis, contenu du message et correspondance
    des identifiants courts."""

    criteria: list[Criterion] = field(default_factory=list)
    content: dict[str, Any] = field(default_factory=dict)
    criterion_keys: dict[str, UUID] = field(default_factory=dict)
    question_keys: dict[str, UUID] = field(default_factory=dict)
    excerpt_keys: dict[str, UUID] = field(default_factory=dict)
    excerpts: dict[str, Excerpt] = field(default_factory=dict)
    # Extraits proposés pour chaque sous-question : seuls citables (GF-1).
    allowed: dict[UUID, frozenset[str]] = field(default_factory=dict)
    # Extraits citables par la justification d'un critère : ceux de ses
    # sous-questions.
    criterion_allowed: dict[UUID, frozenset[str]] = field(default_factory=dict)
    # Langues de traduction demandées (`tr`), hors langue de rédaction.
    locales: tuple[str, ...] = ()
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0

    def message(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class Grading:
    """Notes obtenues de l'IA, par sous-question et par critère."""

    graded_criteria: set[UUID] = field(default_factory=set)
    grades: dict[UUID, Grade] = field(default_factory=dict)
    # Justification sourcée et traduite de chaque critère.
    justifications: dict[UUID, Reading] = field(default_factory=dict)


@dataclass(slots=True)
class SubAnswerResult:
    id: UUID
    sub_question_id: UUID
    status: AnswerStatus
    note: int | None
    contradictory: bool
    evidences: list[Candidate]
    # Critère de la sous-question : la couche C y rattache ses preuves.
    criterion_id: UUID | None = None
    # Par segment cité : positions des phrases citées.
    spans: Mapping[UUID, tuple[Span, ...]] = field(default_factory=dict)
    interpretation: Reading | None = None


@dataclass(slots=True)
class CriterionNote:
    criterion_id: UUID
    note: float | None
    status: NoteStatus
    documented: int
    total: int
    justification: str
    ambiguous: bool
    # Justification rédigée par l'IA : sources numérotées (« ⟦1⟧ ») et
    # traductions ; `None` pour un libellé factuel.
    written: Reading | None = None


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
                    "justification_translations": (
                        note.written.translations if note.written else {}
                    ),
                    "citations": [
                        source.as_dict()
                        for source in (note.written.citations if note.written else [])
                    ],
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
                    **reading_payload(answer.interpretation),
                }
                for answer in self.answers
            ],
            # La lecture de chaque extrait vient de l'étape de lecture
            # (`readings`), commune à toutes les couches.
            "evidences": [
                {
                    "sub_answer_id": str(answer.id),
                    "segment_id": str(candidate.segment_id),
                    "similarity": candidate.similarity,
                    "spans": (
                        [list(span) for span in answer.spans[candidate.segment_id]]
                        if answer.spans.get(candidate.segment_id)
                        else None
                    ),
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
    excerpts: dict[str, list[str]] = dict(batch.content.get("x", {}))
    excerpt_items = dict(batch.excerpts)
    excerpt_keys = dict(batch.excerpt_keys)
    key_of_segment = {segment: key for key, segment in excerpt_keys.items()}
    question_keys = dict(batch.question_keys)
    allowed = dict(batch.allowed)
    criterion_keys = dict(batch.criterion_keys)
    criterion_allowed = dict(batch.criterion_allowed)
    locales = tuple(other for other in INTERFACE_LOCALES if other != inputs.locale)

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
                excerpt_items[key] = excerpt(
                    candidate.segment_id, candidate.text, rules.excerpt_chars
                )
                excerpts[key] = list(excerpt_items[key].sentences)
            keys.append(key)
        allowed[question.id] = frozenset(keys)
        item: dict[str, Any] = {"k": question_key, "t": question.text, "e": keys}
        if question.expected_evidence:
            item["p"] = truncate(question.expected_evidence, EXPECTED_EVIDENCE_CHARS)
        asked.append(item)
    criterion_allowed[criterion.id] = frozenset(
        key for question in questions for key in allowed[question.id]
    )

    entry: dict[str, Any] = {
        "k": criterion_key,
        "n": criterion.name,
        "s": criterion.applicability,
        "q": asked,
    }
    if criterion.definition:
        entry["d"] = truncate(criterion.definition, DEFINITION_CHARS)
    content: dict[str, Any] = {"lang": inputs.locale, "cycle": inputs.cycle}
    if project := inputs.project.prompt():
        content["pj"] = project
    if locales:
        content["tr"] = list(locales)
    content |= {
        "x": excerpts,
        "c": [*batch.content.get("c", []), entry],
    }
    extended = GradingBatch(
        criteria=[*batch.criteria, criterion],
        content=content,
        criterion_keys=criterion_keys,
        question_keys=question_keys,
        excerpt_keys=excerpt_keys,
        excerpts=excerpt_items,
        allowed=allowed,
        criterion_allowed=criterion_allowed,
        locales=locales,
    )
    extended.estimated_input_tokens = SYSTEM_PROMPT_TOKENS + (
        len(extended.message()) // CHARS_PER_TOKEN
    )
    # Chaque traduction réécrit interprétations et justifications.
    extended.max_output_tokens = (1 + len(locales)) * (
        OUTPUT_TOKENS_PER_QUESTION * len(question_keys)
        + OUTPUT_TOKENS_PER_CRITERION * len(criterion_keys)
    )
    return extended


def interpret(batch: GradingBatch, raw: RawGrading, grading: Grading) -> None:
    """Ajoute à `grading` les notes valides d'une réponse IA. Une note hors
    barème, sans extrait cité, ou citant un extrait qui n'était pas proposé
    pour la question, n'est pas retenue : la question reste non documentée.
    L'interprétation d'une sous-question cite ses propres extraits, celle
    d'un critère les extraits de ses sous-questions ; trop courte ou sans
    source, elle n'est pas publiée."""
    grading.graded_criteria.update(criterion.id for criterion in batch.criteria)
    translations = {
        locale: raw.translations[locale]
        for locale in batch.locales
        if locale in raw.translations
    }
    for raw_grade in raw.grades:
        question_id = batch.question_keys.get(raw_grade.key)
        if question_id is None or question_id in grading.grades:
            continue
        if raw_grade.note is None or raw_grade.note not in (0, 1, 2, 3):
            continue
        allowed = batch.allowed.get(question_id, frozenset())
        found = cited(raw_grade.excerpts, batch.excerpts, allowed)
        if not found:
            continue
        interpretation = written(
            raw_grade.interpretation,
            {
                locale: translation.interpretations.get(raw_grade.key)
                for locale, translation in translations.items()
            },
            batch.excerpts,
            allowed,
            INTERPRETATION_CHARS,
            INTERPRETATION_MAX_CHARS,
            INTERPRETATION_MIN_CHARS,
        )
        grading.grades[question_id] = Grade(
            raw_grade.note,
            tuple(batch.excerpt_keys[key] for key in found),
            raw_grade.contradictory,
            spans={
                batch.excerpt_keys[key]: tuple(merge(spans))
                for key, spans in found.items()
            },
            interpretation=interpretation,
        )
    for key, text in raw.justifications.items():
        criterion_id = batch.criterion_keys.get(key)
        if criterion_id is None or criterion_id in grading.justifications:
            continue
        justification_text = written(
            text,
            {
                locale: translation.justifications.get(key)
                for locale, translation in translations.items()
            },
            batch.excerpts,
            batch.criterion_allowed.get(criterion_id, ()),
            JUSTIFICATION_CHARS,
            JUSTIFICATION_MAX_CHARS,
            JUSTIFICATION_MIN_CHARS,
        )
        if justification_text is not None:
            grading.justifications[criterion_id] = justification_text


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
                    criterion_id=criterion.id,
                    spans=grade.spans if grade else {},
                    interpretation=grade.interpretation if grade else None,
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
        written_justification = grading.justifications.get(criterion.id)
        notes.append(
            CriterionNote(
                criterion_id=criterion.id,
                note=round(statistics.fmean(grade.note for grade in grades), 1),
                status="conclu",
                documented=documented,
                total=total,
                justification=written_justification.text
                if written_justification
                else justification(inputs.locale, reason, documented, total),
                ambiguous=is_ambiguous(grades, rules.ambiguity_std),
                written=written_justification,
            )
        )
    return LayerBResults(notes=notes, answers=answers)
