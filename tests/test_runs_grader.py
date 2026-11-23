from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import httpx
import openai
import pytest

from app.core.config import Settings
from app.modules.runs import grader as grader_module
from app.modules.runs.domain import LayerAResults
from app.modules.runs.grader import (
    GradingUnavailable,
    LlmAlert,
    LlmAlertMessage,
    LlmAlertTranslation,
    LlmCriterionJustification,
    LlmGrade,
    LlmGradingOutput,
    LlmGradingTranslation,
    LlmProjection,
    LlmQualificationOutput,
    LlmReadingOutput,
    LlmReadingTranslation,
    LlmRestitutionAction,
    LlmRestitutionItem,
    LlmRestitutionOutput,
    LlmRestitutionTranslation,
    LlmTranslatedAction,
    LlmTranslatedItem,
    OpenAIAlertQualifier,
    OpenAIEvidenceReader,
    OpenAIRestitutionWriter,
    OpenAISubQuestionGrader,
    build_alert_qualifier,
    build_evidence_reader,
    build_restitution_writer,
    build_sub_question_grader,
)
from app.modules.runs.layer_b import (
    Candidate,
    Criterion,
    LayerBInputs,
    LayerBRules,
    SubQuestion,
    build_batches,
    retrieve,
)
from app.modules.runs.layer_c import (
    RULES,
    Alert,
    LayerCInputs,
    LayerCRules,
    PillarCriterion,
    RawQualification,
)
from app.modules.runs.layer_c import build_batches as build_alert_batches
from app.modules.runs.reading import ReadingBatch
from app.modules.runs.restitution import (
    RawProjection,
    RestitutionRules,
    build_batch,
    select,
)

PROMPT = {
    "code": "grade_sub_questions",
    "version": "v1.0",
    "template": "Noter les sous-questions.",
    "model": "test-llm",
    "parameters": {"temperature": 0, "max_tokens": 6000},
}


class FakeResponses:
    def __init__(self, output=None, error: Exception | None = None, status=None):
        self._output = output
        self._error = error
        self._status = status
        self.last_request = None

    async def parse(self, **kwargs):
        self.last_request = kwargs
        if self._error:
            raise self._error
        return SimpleNamespace(
            output_parsed=self._output,
            status=self._status,
            usage=SimpleNamespace(input_tokens=2000, output_tokens=100),
        )


class FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.responses = FakeResponses(**kwargs)


def _batch():
    question = SubQuestion(uuid4(), "Les besoins sont-ils analysés ?")
    criterion = Criterion(
        uuid4(), "pertinence", "Pertinence", None, "obligatoire", (question,)
    )
    rules = LayerBRules(min_similarity=0.4)
    retrieved = retrieve(
        [Candidate(question.id, uuid4(), uuid4(), 0.6, "Analyse des besoins.")], rules
    )
    (batch,) = build_batches(
        LayerBInputs("fr", "finale", (criterion,)), retrieved, rules
    )
    return batch


def _settings(**overrides) -> Settings:
    return Settings(
        ai_provider="openai",
        openai_api_key="test",
        openai_input_cost_eur_per_million=Decimal("2"),
        openai_output_cost_eur_per_million=Decimal("8"),
        **overrides,
    )


OUTPUT = LlmGradingOutput(
    q=[
        LlmGrade(
            k="Q1",
            n=2,
            e=["E1.1"],
            c=False,
            t="Le projet répond aux priorités.",
        )
    ],
    c=[LlmCriterionJustification(k="C1", j="Besoins analysés.")],
    tr=[
        LlmGradingTranslation(
            l="en",
            q=[LlmTranslatedItem(k="Q1", t="The project meets the priorities.")],
            c=[LlmCriterionJustification(k="C1", j="Needs analysed.")],
        )
    ],
)


async def test_one_structured_call_returns_the_ratings_and_their_cost() -> None:
    client = FakeOpenAI(output=OUTPUT)
    batch = _batch()

    raw, usage = await OpenAISubQuestionGrader(_settings(), client).grade(PROMPT, batch)

    request = client.responses.last_request
    assert request["model"] == "test-llm"
    assert request["text_format"] is LlmGradingOutput
    # Le prompt porte temperature, refusée par le modèle : jamais envoyée.
    assert "temperature" not in request
    assert request["max_output_tokens"] == 6000
    assert request["input"][0]["content"] == "Noter les sous-questions."
    assert request["input"][1]["content"] == batch.message()
    assert raw.grades[0].key == "Q1" and raw.grades[0].note == 2
    assert raw.grades[0].excerpts == ("E1.1",)
    assert raw.grades[0].interpretation == "Le projet répond aux priorités."
    assert raw.justifications == {"C1": "Besoins analysés."}
    translation = raw.translations["en"]
    assert translation.interpretations == {"Q1": "The project meets the priorities."}
    assert translation.justifications == {"C1": "Needs analysed."}
    assert usage.cost_eur == Decimal("0.004800")


async def test_the_estimated_cost_is_checked_before_calling() -> None:
    client = FakeOpenAI(output=OUTPUT)
    grader = OpenAISubQuestionGrader(
        _settings(ai_max_cost_eur=Decimal("0.000001")), client
    )

    with pytest.raises(GradingUnavailable) as error:
        await grader.grade(PROMPT, _batch())

    assert error.value.code == "AI_COST_LIMIT_EXCEEDED"
    assert client.responses.last_request is None


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {
                "error": openai.APIConnectionError(
                    request=httpx.Request("POST", "https://api.test")
                )
            },
            "AI_PROVIDER_UNAVAILABLE",
        ),
        ({"output": None}, "AI_PROVIDER_RESPONSE_INVALID"),
        ({"output": OUTPUT, "status": "incomplete"}, "AI_OUTPUT_TRUNCATED"),
    ],
)
async def test_provider_failures_make_the_grading_unavailable(kwargs, code) -> None:
    grader = OpenAISubQuestionGrader(_settings(), FakeOpenAI(**kwargs))

    with pytest.raises(GradingUnavailable) as error:
        await grader.grade(PROMPT, _batch())

    assert error.value.code == code


def test_no_grader_when_ai_is_disabled() -> None:
    assert build_sub_question_grader(Settings(ai_provider="disabled")) is None
    assert build_sub_question_grader(_settings()) is not None


def _qualification_batch():
    pillar, criterion, version = uuid4(), uuid4(), uuid4()
    inputs = LayerCInputs(
        "fr",
        {pillar: "Budget"},
        (PillarCriterion(pillar, criterion, "pertinence", "Pertinence"),),
    )
    alert = Alert(
        id=uuid4(),
        rule=RULES["ALERT-01"],
        severity="majeure",
        pillar_id=pillar,
        pillar_name="Budget",
        criterion_id=criterion,
        criterion_name="Pertinence",
        program_version_id=version,
        score=62,
        note=1,
        delta=None,
        confidence=0.8,
        segment_ids=[uuid4()],
        message="Écart.",
    )
    (batch,) = build_alert_batches(
        inputs, [alert], {alert.segment_ids[0]: "Extrait."}, LayerCRules()
    )
    return batch


async def test_one_structured_call_qualifies_the_gaps() -> None:
    output = LlmQualificationOutput(
        a=[LlmAlert(k="A1", ok=True, g="mineure", m="Écart explicable.", e=["E1"])],
        tr=[
            LlmAlertTranslation(
                l="en", a=[LlmAlertMessage(k="A1", m="Explainable gap.")]
            )
        ],
    )
    client = FakeOpenAI(output=output)
    batch = _qualification_batch()
    prompt = {**PROMPT, "parameters": {"max_tokens": 2000}}

    raw, usage = await OpenAIAlertQualifier(_settings(), client).qualify(prompt, batch)

    request = client.responses.last_request
    assert request["text_format"] is LlmQualificationOutput
    assert request["max_output_tokens"] == 2000
    assert "temperature" not in request
    assert request["input"][1]["content"] == batch.message()
    assert raw == [
        RawQualification(
            "A1",
            True,
            "mineure",
            "Écart explicable.",
            ("E1",),
            {"en": "Explainable gap."},
        )
    ]
    assert usage.cost_eur == Decimal("0.004800")


def test_no_qualifier_when_ai_is_disabled() -> None:
    assert build_alert_qualifier(Settings(ai_provider="disabled")) is None
    assert build_alert_qualifier(_settings()) is not None


async def test_one_structured_call_writes_the_restitution() -> None:
    batch = _qualification_batch()
    alert = batch.alerts[0]
    inputs = LayerCInputs("fr", {alert.pillar_id: "Budget"}, ())
    restitution = select(
        inputs, LayerAResults([], []), None, [alert], RestitutionRules()
    )
    restitution_batch = build_batch(
        "fr", restitution, {alert.segment_ids[0]: "Extrait."}, RestitutionRules()
    )
    assert restitution_batch is not None
    projection = LlmProjection(t="Vérifier l'écart.", j="l'écart est majeur")
    english = LlmProjection(t="Check the gap.", j="the gap is major")
    output = LlmRestitutionOutput(
        i=[LlmRestitutionItem(k="V1", t="Écart à vérifier.", e=["E1"])],
        c="Les résultats suggèrent un écart.",
        a=[
            LlmRestitutionAction(
                k="V1", e=["E1"], dec=projection, pmu=projection, ana=projection
            )
        ],
        tr=[
            LlmRestitutionTranslation(
                l="en",
                i=[LlmTranslatedItem(k="V1", t="Gap to verify.")],
                c="Results suggest a gap.",
                a=[LlmTranslatedAction(k="V1", dec=english, pmu=english, ana=english)],
            )
        ],
    )
    client = FakeOpenAI(output=output)
    prompt = {**PROMPT, "parameters": {"max_tokens": 3000}}

    raw, _ = await OpenAIRestitutionWriter(_settings(), client).write(
        prompt, restitution_batch
    )

    request = client.responses.last_request
    assert request["text_format"] is LlmRestitutionOutput
    # Le budget de sortie couvre la traduction demandée (tr) : il dépasse le
    # plafond du prompt.
    assert restitution_batch.content["tr"] == ["en"]
    assert request["max_output_tokens"] == max(
        3000, restitution_batch.max_output_tokens + 500
    )
    assert request["input"][1]["content"] == restitution_batch.message()
    assert raw.conclusion == "Les résultats suggèrent un écart."
    (action,) = raw.actions
    assert set(action.projections) == {"decideur", "pmu", "analyste"}
    assert action.projections["analyste"] == RawProjection(
        "Vérifier l'écart.", "l'écart est majeur"
    )
    translation = raw.translations["en"]
    assert translation.items == {"V1": "Gap to verify."}
    assert translation.conclusion == "Results suggest a gap."
    assert translation.actions["V1"]["pmu"] == RawProjection(
        "Check the gap.", "the gap is major"
    )


def test_no_restitution_writer_when_ai_is_disabled() -> None:
    assert build_restitution_writer(Settings(ai_provider="disabled")) is None
    assert build_restitution_writer(_settings()) is not None


async def test_one_structured_call_reads_extracts_variables_and_pillars() -> None:
    output = LlmReadingOutput(
        s=[LlmTranslatedItem(k="E1", t="Le projet a un comité [E1.1].")],
        v=[LlmTranslatedItem(k="V1", t="Le projet se pilote [E1].")],
        p=[LlmTranslatedItem(k="P1", t="La gouvernance du projet [E1].")],
        tr=[
            LlmReadingTranslation(
                l="en",
                s=[LlmTranslatedItem(k="E1", t="The project has a board [E1.1].")],
                v=[],
                p=[LlmTranslatedItem(k="P1", t="Project governance [E1].")],
            )
        ],
    )
    client = FakeOpenAI(output=output)
    batch = ReadingBatch(content={"lang": "fr"}, max_output_tokens=9000)
    prompt = {**PROMPT, "parameters": {"max_tokens": 4000}}

    raw, _ = await OpenAIEvidenceReader(_settings(), client).read(prompt, batch)

    request = client.responses.last_request
    assert request["text_format"] is LlmReadingOutput
    assert request["max_output_tokens"] == 9500
    assert raw.segments == {"E1": "Le projet a un comité [E1.1]."}
    assert raw.variables == {"V1": "Le projet se pilote [E1]."}
    assert raw.pillars == {"P1": "La gouvernance du projet [E1]."}
    translation = raw.translations["en"]
    assert translation.segments == {"E1": "The project has a board [E1.1]."}
    assert translation.variables == {}
    assert translation.pillars == {"P1": "Project governance [E1]."}


def test_no_evidence_reader_when_ai_is_disabled() -> None:
    assert build_evidence_reader(Settings(ai_provider="disabled")) is None
    assert build_evidence_reader(_settings()) is not None


async def test_temperature_is_never_sent() -> None:
    """gpt-6-astra refuse `temperature` (400) : un prompt qui la porte encore
    ne doit pas faire rejeter toute la couche B."""
    client = FakeOpenAI(output=OUTPUT)

    raw, _ = await OpenAISubQuestionGrader(_settings(), client).grade(
        {**PROMPT, "parameters": {"temperature": 0, "max_tokens": 24000}}, _batch()
    )

    assert "temperature" not in client.responses.last_request
    assert raw.grades[0].note == 2


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(grader_module, "RETRY_DELAY_SECONDS", 0)


class Flaky(FakeResponses):
    """Échoue avec les erreurs données, dans l'ordre, puis répond."""

    def __init__(self, output, errors: list[Exception]) -> None:
        super().__init__(output=output)
        self.errors = errors
        self.requests: list[dict] = []

    async def parse(self, **kwargs):
        self.requests.append(dict(kwargs))
        if self.errors:
            raise self.errors.pop(0)
        return await super().parse(**kwargs)


def _http_request():
    return httpx.Request("POST", "https://api.openai.test/v1/responses")


async def test_the_timeout_follows_the_expected_output() -> None:
    client = FakeOpenAI(output=OUTPUT)
    settings = _settings(
        run_ai_timeout_base_seconds=90,
        run_ai_output_tokens_per_second=25,
        run_ai_timeout_max_seconds=600,
    )
    grader = OpenAISubQuestionGrader(settings, client)
    batch = _batch()

    await grader.grade(PROMPT, batch)

    assert client.responses.last_request["timeout"] == pytest.approx(
        90 + batch.max_output_tokens / 25
    )
    assert grader.timeout(5000) == 290
    assert grader.timeout(100_000) == 600


async def test_a_timeout_is_not_retried() -> None:
    client = FakeOpenAI()
    client.responses = Flaky(OUTPUT, [openai.APITimeoutError(_http_request())])

    with pytest.raises(GradingUnavailable) as error:
        await OpenAISubQuestionGrader(_settings(), client).grade(PROMPT, _batch())

    assert error.value.code == "AI_PROVIDER_UNAVAILABLE"
    assert "n'a pas répondu" in error.value.message
    assert len(client.responses.requests) == 1


async def test_a_passing_incident_is_retried_once() -> None:
    client = FakeOpenAI()
    client.responses = Flaky(
        OUTPUT,
        [
            openai.RateLimitError(
                "slow down",
                response=httpx.Response(429, request=_http_request()),
                body=None,
            )
        ],
    )

    raw, _ = await OpenAISubQuestionGrader(_settings(), client).grade(PROMPT, _batch())

    assert len(client.responses.requests) == 2
    assert raw.grades[0].note == 2


async def test_a_lasting_incident_gives_up_after_one_retry() -> None:
    client = FakeOpenAI()
    client.responses = Flaky(
        OUTPUT,
        [openai.APIConnectionError(request=_http_request()) for _ in range(3)],
    )

    with pytest.raises(GradingUnavailable) as error:
        await OpenAISubQuestionGrader(_settings(), client).grade(PROMPT, _batch())

    assert error.value.code == "AI_PROVIDER_UNAVAILABLE"
    assert len(client.responses.requests) == 2
