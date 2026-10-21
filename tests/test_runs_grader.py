from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import httpx
import openai
import pytest

from app.core.config import Settings
from app.modules.runs.grader import (
    GradingUnavailable,
    LlmCriterionJustification,
    LlmGrade,
    LlmGradingOutput,
    OpenAISubQuestionGrader,
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
    q=[LlmGrade(k="Q1", n=2, e=["E1"], c=False)],
    c=[LlmCriterionJustification(k="C1", j="Besoins analysés.")],
)


async def test_one_structured_call_returns_the_ratings_and_their_cost() -> None:
    client = FakeOpenAI(output=OUTPUT)
    batch = _batch()

    raw, usage = await OpenAISubQuestionGrader(_settings(), client).grade(PROMPT, batch)

    request = client.responses.last_request
    assert request["model"] == "test-llm"
    assert request["text_format"] is LlmGradingOutput
    assert request["temperature"] == 0
    assert request["max_output_tokens"] == 6000
    assert request["input"][0]["content"] == "Noter les sous-questions."
    assert request["input"][1]["content"] == batch.message()
    assert raw.grades[0].key == "Q1" and raw.grades[0].note == 2
    assert raw.justifications == {"C1": "Besoins analysés."}
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
