"""Rapport d'un run (E14, écran 22), commun à tous les formats d'export.

`build_report` assemble, à partir des lignes lues en base, ce que chaque
format met en page, dans l'ordre des écrans (US-22.2) : comparatif, piliers
(couche A), critères (couche B), alertes (couche C), synthèse et plan
d'action. Les trois couches restent distinctes (US-22.3).

Les limites du corpus ouvrent toujours le rapport (US-22.5) : niveau de
masse critique, critères non conclus, confiance, variables non renseignées,
couches non calculées, corpus modifié, absence de validation. Aucun format
ne peut produire un rapport « propre » qui les masque. Un run exploratoire
porte son bandeau en tête.

Les preuves deviennent des notes numérotées, une par document et par page
(US-22.4), dans l'ordre de leur première citation ; chaque élément renvoie à
ses numéros, regroupés en plages. Une note cite les extraits les plus
proches de ce qu'ils étayent (similarité), deux au plus.

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from app.modules.exports.labels import labels_for

EXCERPT_CHARS = 400
EXCERPTS_PER_NOTE = 2
Persona = str


@dataclass(frozen=True, slots=True)
class RunRows:
    """Lignes brutes du run, telles que PostgREST les renvoie."""

    run: dict[str, Any]
    steps: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    variables: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    evidences: list[dict[str, Any]]
    synthesis: dict[str, Any] | None
    items: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    corpus_changed: bool


@dataclass(frozen=True, slots=True)
class Note:
    """Une page d'un document, citée par un ou plusieurs éléments."""

    number: int
    document: str | None
    page: int | None
    section: str | None
    excerpts: list[str]
    subjects: list[str]
    #: Extraits de cette page retenus par le moteur, cités ou non.
    extracts: int

    @property
    def excerpt(self) -> str:
        return "\n".join(self.excerpts)

    @property
    def subject(self) -> str:
        return " ; ".join(self.subjects)


@dataclass(frozen=True, slots=True)
class PillarScore:
    version: str
    version_order: int
    score: float | None
    delta: float | None
    evolution: str | None
    confidence: float
    documents: int


@dataclass(frozen=True, slots=True)
class VariableLine:
    code: str
    label: str
    version: str
    state: str
    value: float | None
    documents: int
    confidence: float | None
    notes: list[int]


@dataclass(frozen=True, slots=True)
class PillarResult:
    id: str
    name: str
    description: str
    scores: list[PillarScore]
    variables: list[VariableLine]
    notes: list[int]

    @property
    def latest(self) -> PillarScore:
        return self.scores[-1]


@dataclass(frozen=True, slots=True)
class CriterionResult:
    id: str
    code: str
    name: str
    note: float | None
    status: str
    documented: int
    total: int
    justification: str | None
    ambiguous: bool
    notes: list[int]


@dataclass(frozen=True, slots=True)
class AlertResult:
    id: str
    rule: str
    type: str
    severity: str
    message: str
    qualified: bool
    status: str
    comment: str | None
    scope: str
    notes: list[int]


@dataclass(frozen=True, slots=True)
class SynthesisLine:
    section: str
    position: int
    text: str
    origin: str
    notes: list[int]


@dataclass(frozen=True, slots=True)
class ActionLine:
    position: int
    text: str
    justification: str
    origin: str
    notes: list[int]


@dataclass(frozen=True, slots=True)
class Report:
    locale: str
    persona: Persona
    workspace: str
    run_id: str
    run_date: str
    engine_version: str
    corpus_size: int
    corpus_level: str
    validated_by: str | None
    validated_at: str | None
    exported_at: datetime
    exported_by: str
    pillars: list[PillarResult]
    criteria: list[CriterionResult]
    alerts: list[AlertResult]
    synthesis: list[SynthesisLine]
    conclusion: str | None
    restitution_status: str | None
    actions: list[ActionLine]
    limits: list[str]
    notes: list[Note]
    layers: dict[str, bool] = field(default_factory=dict)

    @property
    def exploratory(self) -> bool:
        return self.corpus_level == "exploratoire"

    @property
    def labels(self) -> dict[str, Any]:
        return labels_for(self.locale)

    @property
    def average_score(self) -> float | None:
        values = [
            pillar.latest.score
            for pillar in self.pillars
            if pillar.latest.score is not None
        ]
        return round(sum(values) / len(values)) if values else None

    @property
    def multiple_versions(self) -> bool:
        return any(len(pillar.scores) > 1 for pillar in self.pillars)

    # --- Mise en forme, commune aux formats ---------------------------------

    def number(self, value: float | None, digits: int = 0) -> str:
        if value is None:
            return self.labels["empty"]
        text = f"{value:.{digits}f}"
        return text.replace(".", ",") if self.locale == "fr" else text

    def signed(self, value: float | None) -> str:
        if value is None:
            return self.labels["empty"]
        return ("+" if value > 0 else "") + self.number(value)

    def percent(self, value: float | None) -> str:
        if value is None:
            return self.labels["empty"]
        separator = "\u202f" if self.locale == "fr" else ""
        return f"{round(value * 100)}{separator}%"

    def date(self, value: str | datetime | None, *, time: bool = False) -> str:
        if value is None:
            return self.labels["empty"]
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        pattern = "%d/%m/%Y" if self.locale == "fr" else "%Y-%m-%d"
        return moment.strftime(f"{pattern} %H:%M UTC" if time else pattern)

    def refs(self, numbers: list[int]) -> str:
        """Numéros de notes, les suites regroupées : [3] [7–12] [15]."""
        runs: list[list[int]] = []
        for number in sorted(set(numbers)):
            if runs and number == runs[-1][-1] + 1:
                runs[-1].append(number)
            else:
                runs.append([number])
        return " ".join(
            f"[{run[0]}–{run[-1]}]"
            if len(run) > 2
            else " ".join(f"[{number}]" for number in run)
            for run in runs
        )

    def omitted(self, note: Note) -> str | None:
        """Mention des extraits non cités, quand la note en écarte."""
        if note.extracts <= len(note.excerpts):
            return None
        return self.labels["extracts"].format(
            shown=len(note.excerpts), count=note.extracts
        )

    def labeled(self, name: str, value: object) -> str:
        separator = " : " if self.locale == "fr" else ": "
        return f"{name}{separator}{value}"

    def quote(self, text: str) -> str:
        return f"« {text} »" if self.locale == "fr" else f"“{text}”"

    def score(self, value: float | None) -> str:
        return self.labels["not_computed"] if value is None else self.number(value)

    @property
    def notes_intro(self) -> str:
        return self.labels["notes_intro"].format(
            date=self.date(self.run_date), engine=self.engine_version
        )

    @property
    def validation(self) -> str:
        meta = self.labels["meta"]
        if not self.validated_at:
            return meta["not_validated"]
        return meta["validated"].format(
            name=self.validated_by or meta["unknown_member"],
            date=self.date(self.validated_at),
        )

    @property
    def corpus(self) -> str:
        return self.labels["meta"]["corpus_value"].format(
            size=self.corpus_size,
            level=self.labels["levels"].get(self.corpus_level, self.corpus_level),
            scope=self.labels["scope"].get(self.corpus_level, ""),
        )

    @property
    def exploratory_text(self) -> str:
        return self.labels["exploratory"]["text"].format(count=self.corpus_size)

    def meta_rows(self) -> list[tuple[str, str]]:
        meta = self.labels["meta"]
        return [
            (meta["workspace"], self.workspace),
            (meta["run"], self.run_id),
            (meta["run_date"], self.date(self.run_date, time=True)),
            (meta["engine"], self.engine_version),
            (meta["corpus"], self.corpus),
            (meta["validation"], self.validation),
            (meta["persona"], self.labels["personas"].get(self.persona, "")),
            (
                meta["exported"],
                meta["exported_value"].format(
                    date=self.date(self.exported_at, time=True),
                    author=self.exported_by,
                ),
            ),
        ]

    def actions_empty(self) -> str:
        empty = self.labels["actions_empty"]
        return empty.get(self.restitution_status or "missing", empty["missing"])


def localized(row: dict[str, Any], locale: str, translations: str) -> dict[str, Any]:
    """Texte dans la langue demandée, sinon dans sa langue source
    (même règle que le dashboard)."""
    if row.get("source_locale") == locale:
        return row
    for translation in row.get(translations) or []:
        if translation.get("locale") == locale:
            return {**row, **{k: v for k, v in translation.items() if v}}
    return row


@dataclass(slots=True)
class _Page:
    number: int
    first: dict[str, Any]
    rows: list[dict[str, Any]]
    subjects: list[str]


class _Citations:
    """Numérote les pages citées dans l'ordre de leur première citation."""

    def __init__(self, evidences: dict[str, dict[str, Any]], subject) -> None:
        self._evidences = evidences
        self._subject = subject
        self._pages: dict[tuple[str, Any], _Page] = {}

    @staticmethod
    def _key(row: dict[str, Any]) -> tuple[str, Any]:
        # Sans document (supprimé depuis le run), l'extrait reste seul.
        if not row.get("document_id"):
            return ("evidence", row["id"])
        return (row["document_id"], row.get("page_number"))

    def cite(self, evidence_ids: list[str]) -> list[int]:
        numbers: set[int] = set()
        for evidence_id in evidence_ids:
            row = self._evidences.get(evidence_id)
            if row is None:
                continue
            key = self._key(row)
            page = self._pages.get(key)
            if page is None:
                page = _Page(len(self._pages) + 1, row, [], [])
                self._pages[key] = page
            if all(item["id"] != row["id"] for item in page.rows):
                page.rows.append(row)
            subject = self._subject(row)
            if subject not in page.subjects:
                page.subjects.append(subject)
            numbers.add(page.number)
        return sorted(numbers)

    @property
    def notes(self) -> list[Note]:
        notes = []
        for page in self._pages.values():
            ranked = sorted(
                page.rows,
                key=lambda row: -float(row.get("similarity") or 0),
            )
            excerpts: list[str] = []
            for row in ranked:
                excerpt = " ".join(str(row.get("excerpt") or "").split())
                if len(excerpt) > EXCERPT_CHARS:
                    excerpt = excerpt[: EXCERPT_CHARS - 1].rstrip() + "…"
                if excerpt and excerpt not in excerpts:
                    excerpts.append(excerpt)
                if len(excerpts) == EXCERPTS_PER_NOTE:
                    break
            notes.append(
                Note(
                    number=page.number,
                    document=(page.first.get("documents") or {}).get(
                        "original_filename"
                    ),
                    page=page.first.get("page_number"),
                    section=next(
                        (
                            row["section_title"]
                            for row in page.rows
                            if row.get("section_title")
                        ),
                        None,
                    ),
                    excerpts=excerpts,
                    subjects=page.subjects,
                    extracts=len(page.rows),
                )
            )
        return notes


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _evidence_ids(row: dict[str, Any], table: str) -> list[str]:
    return [link["evidence_id"] for link in row.get(table) or []]


def _layer_available(steps: list[dict[str, Any]], name: str) -> bool:
    step = next((item for item in steps if item.get("step_name") == name), None)
    return step is None or step.get("status") != "skipped"


def build_report(
    rows: RunRows,
    *,
    locale: str,
    persona: Persona,
    exported_by: str,
    exported_at: datetime,
) -> Report:
    texts = labels_for(locale)
    run = rows.run

    # --- Référentiels du run : noms des piliers, critères et variables ------
    pillar_names: dict[str, str] = {}
    pillar_descriptions: dict[str, str] = {}
    for row in rows.scores + rows.alerts:
        pillar = row.get("pillars")
        if pillar and row.get("pillar_id"):
            text = localized(pillar, locale, "pillar_translations")
            pillar_names[row["pillar_id"]] = text["name"]
            pillar_descriptions[row["pillar_id"]] = text.get("description") or ""
    criterion_names = {
        row["criterion_id"]: (row.get("criteria") or {}).get("name") or "—"
        for row in rows.notes + rows.alerts
        if row.get("criterion_id")
    }
    variable_labels: dict[tuple[str, str], str] = {}
    for row in rows.variables:
        for variable in (row.get("pillars") or {}).get("pillar_variables") or []:
            text = localized(variable, locale, "pillar_variable_translations")
            variable_labels[(row["pillar_id"], variable["code"])] = text["label"]

    def subject(evidence: dict[str, Any]) -> str:
        if evidence.get("criterion_id"):
            return criterion_names.get(evidence["criterion_id"], "—")
        pillar = pillar_names.get(evidence.get("pillar_id") or "", "—")
        code = evidence.get("variable_code")
        if code:
            label = variable_labels.get((evidence.get("pillar_id"), code), code)
            return f"{pillar} · {label}"
        return pillar

    citations = _Citations({row["id"]: row for row in rows.evidences}, subject)

    # --- Couche A : un bloc par pilier, versions dans l'ordre ---------------
    scores_by_pillar: dict[str, list[PillarScore]] = {}
    for row in rows.scores:
        version = row.get("program_versions") or {}
        scores_by_pillar.setdefault(row["pillar_id"], []).append(
            PillarScore(
                version=version.get("label") or "—",
                version_order=int(version.get("order_index") or 0),
                score=_number(row.get("score")),
                delta=_number(row.get("delta")),
                evolution=row.get("evolution_label"),
                confidence=float(row.get("corpus_confidence") or 0),
                documents=int(row.get("documents_count") or 0),
            )
        )
    pillars: list[PillarResult] = []
    for pillar_id in sorted(
        scores_by_pillar, key=lambda key: pillar_names.get(key, "")
    ):
        variables = []
        for row in sorted(
            (item for item in rows.variables if item["pillar_id"] == pillar_id),
            key=lambda item: (
                item["variable_code"],
                (item.get("program_versions") or {}).get("order_index") or 0,
            ),
        ):
            code = row["variable_code"]
            variables.append(
                VariableLine(
                    code=code,
                    label=variable_labels.get((pillar_id, code), code),
                    version=(row.get("program_versions") or {}).get("label") or "—",
                    state=row["state"],
                    value=_number(row.get("value")),
                    documents=int(row.get("documents_count") or 0),
                    confidence=_number(row.get("corpus_confidence")),
                    notes=citations.cite(
                        [
                            evidence["id"]
                            for evidence in rows.evidences
                            if evidence.get("pillar_id") == pillar_id
                            and evidence.get("variable_code") == code
                            and not evidence.get("criterion_id")
                        ]
                    ),
                )
            )
        pillars.append(
            PillarResult(
                id=pillar_id,
                name=pillar_names.get(pillar_id, "—"),
                description=pillar_descriptions.get(pillar_id, ""),
                scores=sorted(
                    scores_by_pillar[pillar_id], key=lambda score: score.version_order
                ),
                variables=variables,
                notes=citations.cite(
                    [
                        evidence["id"]
                        for evidence in rows.evidences
                        if evidence.get("pillar_id") == pillar_id
                        and not evidence.get("variable_code")
                        and not evidence.get("criterion_id")
                    ]
                ),
            )
        )

    # --- Couche B ------------------------------------------------------------
    criteria = [
        CriterionResult(
            id=row["criterion_id"],
            code=(row.get("criteria") or {}).get("code") or "",
            name=criterion_names[row["criterion_id"]],
            note=_number(row.get("note")),
            status=row["status"],
            documented=int(row.get("documented_sub_questions") or 0),
            total=int(row.get("total_sub_questions") or 0),
            justification=row.get("justification"),
            ambiguous=bool(row.get("ambiguity_flag")),
            notes=citations.cite(
                [
                    evidence["id"]
                    for evidence in rows.evidences
                    if evidence.get("criterion_id") == row["criterion_id"]
                ]
            ),
        )
        for row in sorted(
            rows.notes,
            key=lambda item: ((item.get("criteria") or {}).get("code") or "",),
        )
    ]

    # --- Couche C : majeures et à instruire d'abord -------------------------
    alert_scopes: dict[str, str] = {}
    alerts: list[AlertResult] = []
    for row in sorted(
        rows.alerts,
        key=lambda item: (
            item.get("severity") != "majeure",
            item.get("status") != "a_instruire",
            item.get("rule_code") or "",
        ),
    ):
        scope = " · ".join(
            name
            for name in (
                pillar_names.get(row.get("pillar_id") or ""),
                criterion_names.get(row.get("criterion_id") or ""),
            )
            if name
        )
        alert_scopes[row["id"]] = scope
        alerts.append(
            AlertResult(
                id=row["id"],
                rule=row.get("rule_code") or "",
                type=row.get("alert_type") or "",
                severity=row["severity"],
                message=row.get("message") or "",
                qualified=bool(row.get("qualified")),
                status=row["status"],
                comment=row.get("instruction_comment"),
                scope=scope or "—",
                notes=citations.cite(_evidence_ids(row, "alert_evidences")),
            )
        )

    # --- Restitution ---------------------------------------------------------
    def origin(row: dict[str, Any], kind: str) -> str:
        if kind == "alert":
            name = alert_scopes.get(row.get("alert_id") or "")
        elif kind == "criterion":
            name = criterion_names.get(row.get("criterion_id") or "")
        else:
            name = pillar_names.get(row.get("pillar_id") or "")
        separator = " : " if locale == "fr" else ": "
        return f"{texts['origin'][kind]}{separator}{name or '—'}"

    def kind_of(row: dict[str, Any]) -> str:
        return (
            "alert"
            if row.get("alert_id")
            else "criterion"
            if row.get("criterion_id")
            else "pillar"
        )

    synthesis = [
        SynthesisLine(
            section=row["section"],
            position=int(row["position"]),
            text=(row.get("translations") or {}).get(locale) or row["text"],
            origin=origin(row, kind_of(row)),
            notes=citations.cite(_evidence_ids(row, "synthesis_item_evidences")),
        )
        for row in sorted(
            rows.items,
            key=lambda item: (
                ["progres", "faiblesse", "a_verifier"].index(item["section"]),
                item["position"],
            ),
        )
    ]
    actions: list[ActionLine] = []
    for row in sorted(rows.actions, key=lambda item: item["position"]):
        projection = next(
            (
                item
                for item in row.get("run_action_projections") or []
                if item.get("persona") == persona
            ),
            None,
        )
        if projection is None:
            continue
        # Traduction du run (prompt v1.2) si elle existe dans la langue
        # demandée, sinon la langue de rédaction.
        projection = (projection.get("translations") or {}).get(locale) or projection
        actions.append(
            ActionLine(
                position=int(row["position"]),
                text=projection["text"],
                justification=projection["justification"],
                origin=origin(row, row.get("origin_type") or kind_of(row)),
                notes=citations.cite(_evidence_ids(row, "run_action_evidences")),
            )
        )

    layers = {
        "b": _layer_available(rows.steps, "layer_b"),
        "c": _layer_available(rows.steps, "layer_c"),
        "restitution": rows.synthesis is not None,
    }
    report = Report(
        locale=locale,
        persona=persona,
        workspace=(run.get("workspaces") or {}).get("name") or "—",
        run_id=run["id"],
        run_date=run.get("completed_at") or run["created_at"],
        engine_version=run.get("engine_version") or "—",
        corpus_size=int(run.get("corpus_size") or 0),
        corpus_level=run.get("corpus_seuil") or "exploratoire",
        validated_by=run.get("validated_by_name"),
        validated_at=run.get("validated_at"),
        exported_at=exported_at,
        exported_by=exported_by,
        pillars=pillars,
        criteria=criteria,
        alerts=alerts,
        synthesis=synthesis,
        conclusion=(
            ((rows.synthesis or {}).get("translations") or {})
            .get(locale, {})
            .get("conclusion")
            or (rows.synthesis or {}).get("conclusion")
        ),
        restitution_status=(rows.synthesis or {}).get("status"),
        actions=actions,
        limits=[],
        notes=citations.notes,
        layers=layers,
    )
    return replace(report, limits=_limits(report, rows))


def _limits(report: Report, rows: RunRows) -> list[str]:
    """Limites méthodologiques, toujours exportées (US-22.5)."""
    texts = report.labels["limits"]
    lines = [
        texts["level"].format(
            level=report.labels["levels"].get(report.corpus_level, ""),
            scope=report.labels["scope"].get(report.corpus_level, ""),
            size=report.corpus_size,
        )
    ]
    others: list[str] = []
    if not report.layers["b"]:
        others.append(texts["layer_b_missing"])
    not_concluded = sum(c.status == "non_conclu" for c in report.criteria)
    if not_concluded:
        others.append(
            texts["not_concluded"].format(
                count=not_concluded, criteria=len(report.criteria)
            )
        )
    ambiguous = sum(criterion.ambiguous for criterion in report.criteria)
    if ambiguous:
        others.append(texts["ambiguous"].format(count=ambiguous))
    limits = (rows.synthesis or {}).get("limits") or {}
    confidence = limits.get("confidence")
    if confidence is None and report.pillars:
        confidence = sum(p.latest.confidence for p in report.pillars) / len(
            report.pillars
        )
    if confidence is not None:
        others.append(
            texts["confidence"].format(confidence=round(float(confidence) * 100))
        )
    unscored = [p.name for p in report.pillars if p.latest.score is None]
    if unscored:
        others.append(texts["unscored"].format(names=", ".join(unscored)))
    latest_variables = [
        variable
        for pillar in report.pillars
        for variable in pillar.variables
        if variable.version == pillar.latest.version
    ]
    undocumented = sum(v.state == "non_renseigne" for v in latest_variables)
    if undocumented:
        others.append(
            texts["variables"].format(count=undocumented, total=len(latest_variables))
        )
    if not report.layers["c"]:
        others.append(texts["layer_c_missing"])
    pending = sum(alert.status == "a_instruire" for alert in report.alerts)
    if pending:
        others.append(texts["pending_alerts"].format(count=pending))
    unqualified = sum(not alert.qualified for alert in report.alerts)
    if unqualified:
        others.append(texts["unqualified"].format(count=unqualified))
    if not report.layers["restitution"]:
        others.append(texts["restitution_missing"])
    elif report.restitution_status in texts["restitution"]:
        others.append(texts["restitution"][report.restitution_status])
    if rows.corpus_changed:
        others.append(texts["corpus_changed"])
    if not report.validated_at:
        others.append(texts["not_validated"])
    return lines + (others or [report.labels["no_limits"]])
