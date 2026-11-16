"""Banc d'évaluation de la recherche des extraits de la couche B.

Un jeu annoté décrit, pour des sous-questions posées à un corpus réel, les
passages qui y répondent. Le banc rejoue la recherche du moteur sur ce corpus
(même texte de requête, même fonction SQL `segment_matches`, même sélection
`retrieve`) et mesure ce qui atteint l'IA : les `excerpts_per_question`
premiers extraits de chaque sous-question.

Un passage pertinent est désigné par une citation courte (`quote`) tirée du
document : un extrait le trouve s'il contient la citation. Les identifiants
de segments changent à chaque réindexation, une citation reste valable tant
que le texte du document ne change pas.

    # Mesurer la recherche actuelle et la comparer à la version 1.3.0
    python -m app.modules.evals.retrieval run evals/retrieval/projet.json

    # Préparer l'annotation : les candidats de chaque sous-question
    python -m app.modules.evals.retrieval annotate evals/retrieval/projet.json \\
        -o annotation.md
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from app.modules.ingestion.repository import vector_literal
from app.modules.runs.layer_b import (
    Candidate,
    Criterion,
    LayerBInputs,
    LayerBRules,
    SubQuestion,
    retrieve,
)

logger = logging.getLogger(__name__)

# Candidats montrés par sous-question pour l'annotation : plus que ce que
# retient le moteur, sans seuil, pour voir aussi ce qu'il manque.
ANNOTATION_TOP_K = 20
ANNOTATION_EXCERPT_CHARS = 700


class DatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Quote:
    document_id: UUID
    text: str


@dataclass(frozen=True, slots=True)
class Question:
    key: str
    criterion: str
    text: str
    expected_evidence: str | None
    # Vide : la sous-question n'a pas de réponse dans le corpus, le moteur ne
    # devrait rien retenir.
    relevant: tuple[Quote, ...]


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    documents: tuple[UUID, ...]
    questions: tuple[Question, ...]


def load_dataset(path: Path) -> Dataset:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"{path} : {error}") from error
    return parse_dataset(raw, str(path))


def parse_dataset(raw: Any, source: str = "jeu") -> Dataset:
    if not isinstance(raw, dict):
        raise DatasetError(f"{source} : objet JSON attendu")
    try:
        documents = tuple(UUID(value) for value in raw["documents"])
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetError(f"{source} : documents (identifiants) invalides") from error
    if not documents:
        raise DatasetError(f"{source} : aucun document")
    questions: list[Question] = []
    for position, item in enumerate(raw.get("questions") or [], start=1):
        key = str(item.get("id") or f"Q{position}")
        text = str(item.get("text") or "").strip()
        criterion = str(item.get("criterion") or "").strip()
        if not text or not criterion:
            raise DatasetError(f"{source} : {key} sans texte ou sans critère")
        relevant: list[Quote] = []
        for quote in item.get("relevant") or []:
            try:
                document_id = UUID(quote["document"])
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetError(
                    f"{source} : {key}, document de citation invalide"
                ) from error
            if document_id not in documents:
                raise DatasetError(f"{source} : {key} cite un document hors du corpus")
            value = normalize(str(quote.get("quote") or ""))
            if len(value) < 12:
                raise DatasetError(
                    f"{source} : {key}, citation trop courte pour être sûre"
                )
            relevant.append(Quote(document_id, value))
        questions.append(
            Question(
                key,
                criterion,
                text,
                (str(item.get("expected_evidence") or "").strip() or None),
                tuple(relevant),
            )
        )
    if not questions:
        raise DatasetError(f"{source} : aucune sous-question")
    if len({question.key for question in questions}) != len(questions):
        raise DatasetError(f"{source} : identifiants de sous-questions en double")
    return Dataset(str(raw.get("name") or source), documents, tuple(questions))


def normalize(text: str) -> str:
    """Comparaison tolérante : casse, espaces, ligatures et césures."""
    value = unicodedata.normalize("NFKC", text).replace("­", "")
    return " ".join(value.casefold().split())


def found_quotes(question: Question, extracts: Sequence[Candidate]) -> set[Quote]:
    return {
        quote
        for quote in question.relevant
        for candidate in extracts
        if candidate.document_id == quote.document_id
        and quote.text in normalize(candidate.text)
    }


def is_relevant(question: Question, candidate: Candidate) -> bool:
    return bool(found_quotes(question, [candidate]))


@dataclass(frozen=True, slots=True)
class Variant:
    """Une configuration de la recherche à mesurer."""

    name: str
    rules: LayerBRules
    # Preuves attendues ajoutées à la requête (layer-b/1.4.0).
    expected_evidence: bool = True
    # Requête aussi cherchée par ses mots (layer-b/1.4.0).
    hybrid: bool = True


def baseline(rules: LayerBRules) -> Variant:
    """La recherche de layer-b/1.3.0 : vecteurs seuls, requête « critère :
    question », extraits les plus proches sans répartition par document."""
    return Variant(
        "1.3.0",
        replace(rules, lexical_min_similarity=None, max_per_document=None),
        expected_evidence=False,
        hybrid=False,
    )


def queries(dataset: Dataset, variant: Variant) -> list[tuple[str, str]]:
    """Texte de requête de chaque sous-question, construit comme le moteur."""
    inputs = LayerBInputs(
        locale="fr",
        cycle="",
        criteria=tuple(
            Criterion(
                id=_key_id(question.key),
                code=question.criterion,
                name=question.criterion,
                definition=None,
                applicability="obligatoire",
                questions=(
                    SubQuestion(
                        _key_id(question.key),
                        question.text,
                        question.expected_evidence
                        if variant.expected_evidence
                        else None,
                    ),
                ),
            )
            for question in dataset.questions
        ),
    )
    keys = {_key_id(question.key): question.key for question in dataset.questions}
    return [(keys[question_id], text) for question_id, text in inputs.queries()]


def _key_id(key: str) -> UUID:
    """Identifiant stable d'une sous-question du jeu (la base ne la connaît
    pas : il ne sert qu'à relier requêtes et candidats)."""
    return uuid5(NAMESPACE_URL, f"themiros-eval:{key}")


@dataclass(slots=True)
class QuestionResult:
    key: str
    sent: int
    relevant_sent: int
    quotes: int
    quotes_found: int
    first_relevant_rank: int | None
    missed: list[str] = field(default_factory=list)


def evaluate_question(
    question: Question, retrieved: Sequence[Candidate], rules: LayerBRules
) -> QuestionResult:
    sent = list(retrieved[: rules.excerpts_per_question])
    found = found_quotes(question, sent)
    first = next(
        (
            rank
            for rank, candidate in enumerate(sent, start=1)
            if is_relevant(question, candidate)
        ),
        None,
    )
    return QuestionResult(
        key=question.key,
        sent=len(sent),
        relevant_sent=sum(is_relevant(question, candidate) for candidate in sent),
        quotes=len(question.relevant),
        quotes_found=len(found),
        first_relevant_rank=first,
        missed=[quote.text for quote in question.relevant if quote not in found],
    )


def summarize(results: Sequence[QuestionResult]) -> dict[str, float | int | None]:
    """Moyennes par sous-question.

    - rappel : part des passages annotés que l'IA reçoit ;
    - trouvées : part des sous-questions dont l'IA reçoit au moins un passage
      pertinent ;
    - précision : part des extraits envoyés qui sont pertinents ;
    - MRR : inverse du rang du premier extrait pertinent ;
    - sans réponse vides : part des sous-questions sans réponse dans le
      corpus pour lesquelles le moteur n'envoie rien ;
    - extraits : nombre moyen d'extraits envoyés.
    """
    answered = [result for result in results if result.quotes]
    unanswered = [result for result in results if not result.quotes]
    with_extracts = [result for result in results if result.sent]

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 3) if values else None

    return {
        "questions": len(results),
        "rappel": mean([result.quotes_found / result.quotes for result in answered]),
        "trouvees": mean(
            [float(result.first_relevant_rank is not None) for result in answered]
        ),
        "precision": mean(
            [result.relevant_sent / result.sent for result in with_extracts]
        ),
        "mrr": mean(
            [
                1 / result.first_relevant_rank if result.first_relevant_rank else 0.0
                for result in answered
            ]
        ),
        "sans_reponse_vides": mean([float(result.sent == 0) for result in unanswered]),
        "extraits": mean([float(result.sent) for result in results]),
    }


# --- Accès à la base et au modèle ------------------------------------------


class Gateway(Protocol):
    async def service_rpc(
        self, function_name: str, *, payload: dict, request_timeout: float = 30
    ) -> Any: ...

    async def service_select(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> list[dict]: ...


class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


async def search(
    gateway: Gateway,
    embedder: Embedder,
    dataset: Dataset,
    variant: Variant,
    *,
    top_k: int,
    batch_size: int = 8,
) -> dict[str, list[Candidate]]:
    """Candidats bruts de chaque sous-question, par clé du jeu."""
    asked = queries(dataset, variant)
    vectors = await embedder.embed([text for _, text in asked])
    by_id = {_key_id(key): key for key, _ in asked}
    candidates: dict[str, list[Candidate]] = {key: [] for key, _ in asked}
    for start in range(0, len(asked), batch_size):
        rows = await gateway.service_rpc(
            "segment_matches",
            payload={
                "p_document_ids": [str(document) for document in dataset.documents],
                "p_model": embedder.model,
                "p_queries": [
                    {
                        "sub_question_id": str(_key_id(key)),
                        **({"text": text} if variant.hybrid else {}),
                        "embedding": vector_literal(vector),
                    }
                    for (key, text), vector in zip(
                        asked[start : start + batch_size],
                        vectors[start : start + batch_size],
                        strict=True,
                    )
                ],
                "p_top_k": top_k,
            },
            request_timeout=120,
        )
        for row in rows or []:
            candidate = Candidate.from_row(row)
            candidates[by_id[candidate.sub_question_id]].append(candidate)
    return candidates


async def evaluate(
    gateway: Gateway,
    embedder: Embedder,
    dataset: Dataset,
    variant: Variant,
    *,
    batch_size: int = 8,
) -> tuple[dict[str, Any], list[QuestionResult]]:
    raw = await search(
        gateway,
        embedder,
        dataset,
        variant,
        top_k=variant.rules.top_k,
        batch_size=batch_size,
    )
    retrieved = retrieve(
        [candidate for found in raw.values() for candidate in found], variant.rules
    )
    results = [
        evaluate_question(
            question, retrieved.get(_key_id(question.key), []), variant.rules
        )
        for question in dataset.questions
    ]
    return summarize(results), results


def report(dataset: Dataset, measured: dict[str, tuple[dict, list]]) -> str:
    names = list(measured)
    metrics = [
        ("rappel", "Rappel (passages annotés reçus par l'IA)"),
        ("trouvees", "Sous-questions avec au moins un passage pertinent"),
        ("precision", "Précision (extraits pertinents / envoyés)"),
        ("mrr", "MRR (rang du premier passage pertinent)"),
        ("sans_reponse_vides", "Sous-questions sans réponse laissées vides"),
        ("extraits", "Extraits envoyés par sous-question"),
    ]
    width = max(len(label) for _, label in metrics)
    lines = [
        f"Jeu : {dataset.name} — {len(dataset.questions)} sous-questions, "
        f"{len(dataset.documents)} documents",
        "",
        " " * width + "  " + "  ".join(f"{name:>8}" for name in names),
    ]
    for key, label in metrics:
        values = []
        for name in names:
            value = measured[name][0][key]
            values.append(f"{'—' if value is None else value:>8}")
        lines.append(f"{label:<{width}}  " + "  ".join(values))
    last = names[-1]
    missed = [result for result in measured[last][1] if result.missed]
    if missed:
        lines += ["", f"Passages non reçus par l'IA ({last}) :"]
        for result in missed:
            for quote in result.missed:
                lines.append(f"  {result.key} : « {quote[:90]} »")
    return "\n".join(lines)


async def annotation_sheet(
    gateway: Gateway,
    embedder: Embedder,
    dataset: Dataset,
    variant: Variant,
    *,
    batch_size: int = 8,
) -> str:
    """Feuille d'annotation : les candidats de chaque sous-question, sans
    seuil, avec les passages déjà annotés repérés."""
    raw = await search(
        gateway,
        embedder,
        dataset,
        variant,
        top_k=ANNOTATION_TOP_K,
        batch_size=batch_size,
    )
    names = await _document_names(gateway, dataset.documents)
    lines = [f"# Annotation — {dataset.name}", ""]
    lines.append(
        "Pour chaque passage qui répond à la sous-question, copier dans le jeu "
        "une citation courte (une phrase) avec l'identifiant de son document. "
        "✓ : passage déjà annoté."
    )
    for question in dataset.questions:
        lines += ["", f"## {question.key} — {question.criterion}", "", question.text]
        if question.expected_evidence:
            lines.append(f"Preuves attendues : {question.expected_evidence}")
        ranked = sorted(raw.get(question.key, []), key=Candidate.rank_key)
        for rank, candidate in enumerate(ranked, start=1):
            mark = "✓ " if is_relevant(question, candidate) else ""
            words = (
                f", mots n° {candidate.lexical_rank}"
                if candidate.lexical_rank is not None
                else ""
            )
            text = " ".join(candidate.text.split())
            if len(text) > ANNOTATION_EXCERPT_CHARS:
                text = text[:ANNOTATION_EXCERPT_CHARS] + "…"
            lines += [
                "",
                f"{rank}. {mark}{names.get(candidate.document_id, '?')} "
                f"(`{candidate.document_id}`) — similarité "
                f"{candidate.similarity:.2f}{words}",
                f"   > {text}",
            ]
    return "\n".join(lines) + "\n"


async def _document_names(
    gateway: Gateway, documents: Sequence[UUID]
) -> dict[UUID, str]:
    rows = await gateway.service_select(
        "documents",
        params={
            "select": "id,original_filename",
            "id": f"in.({','.join(str(document) for document in documents)})",
        },
    )
    return {UUID(row["id"]): row.get("original_filename") or "?" for row in rows}


# --- Ligne de commande ------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.modules.evals.retrieval",
        description="Banc d'évaluation de la recherche des extraits (couche B).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="mesurer la recherche sur un jeu annoté")
    run.add_argument("dataset", type=Path)
    run.add_argument("--json", type=Path, help="écrire aussi le détail en JSON")
    run.add_argument("--top-k", type=int)
    run.add_argument("--excerpts", type=int, help="extraits envoyés à l'IA")
    run.add_argument("--min-similarity", type=float)
    run.add_argument("--lexical-min-similarity", type=float)
    run.add_argument("--max-per-document", type=int)
    run.add_argument(
        "--no-baseline", action="store_true", help="ne pas mesurer la version 1.3.0"
    )
    annotate = commands.add_parser(
        "annotate", help="écrire la feuille d'annotation d'un jeu"
    )
    annotate.add_argument("dataset", type=Path)
    annotate.add_argument("-o", "--output", type=Path, required=True)
    return parser


def current_rules(
    settings: Any, arguments: argparse.Namespace | None = None
) -> LayerBRules:
    """Règles du moteur (configuration courante), modifiables en ligne de
    commande pour essayer un réglage."""
    rules = LayerBRules(
        min_similarity=settings.run_layer_b_min_similarity,
        lexical_min_similarity=settings.run_layer_b_lexical_min_similarity,
        max_per_document=settings.run_layer_b_max_per_document,
        top_k=settings.run_layer_b_top_k,
        excerpts_per_question=settings.run_layer_b_excerpts_per_question,
    )
    if arguments is None:
        return rules
    overrides = {
        "top_k": arguments.top_k,
        "excerpts_per_question": arguments.excerpts,
        "min_similarity": arguments.min_similarity,
        "lexical_min_similarity": arguments.lexical_min_similarity,
        "max_per_document": arguments.max_per_document,
    }
    return replace(
        rules, **{key: value for key, value in overrides.items() if value is not None}
    )


async def _main(arguments: argparse.Namespace) -> int:
    import httpx

    from app.core.config import get_settings
    from app.infrastructure.supabase import SupabaseGateway
    from app.modules.ingestion.analysis.embeddings import build_embedder

    settings = get_settings()
    dataset = load_dataset(arguments.dataset)
    embedder = build_embedder(settings)
    batch_size = settings.run_layer_b_match_batch_size
    async with httpx.AsyncClient() as client:
        gateway = SupabaseGateway(settings, client)
        if arguments.command == "annotate":
            sheet = await annotation_sheet(
                gateway,
                embedder,
                dataset,
                Variant("courant", current_rules(settings)),
                batch_size=batch_size,
            )
            arguments.output.write_text(sheet, encoding="utf-8")
            print(f"Feuille d'annotation écrite : {arguments.output}")
            return 0

        rules = current_rules(settings, arguments)
        variants = [] if arguments.no_baseline else [baseline(rules)]
        variants.append(Variant("courant", rules))
        measured = {
            variant.name: await evaluate(
                gateway, embedder, dataset, variant, batch_size=batch_size
            )
            for variant in variants
        }
    print(report(dataset, measured))
    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "variants": {
                        name: {
                            "rules": asdict(variant.rules),
                            "expected_evidence": variant.expected_evidence,
                            "hybrid": variant.hybrid,
                            "metrics": measured[name][0],
                            "questions": [
                                asdict(result) for result in measured[name][1]
                            ],
                        }
                        for name, variant in ((item.name, item) for item in variants)
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.exit(asyncio.run(_main(_parser().parse_args())))
    except DatasetError as error:
        print(f"Jeu invalide — {error}", file=sys.stderr)
        sys.exit(2)
