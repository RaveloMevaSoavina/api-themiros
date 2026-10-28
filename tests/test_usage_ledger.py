import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.modules.usage.ledger import UsageLedger, count_embedding

WORKSPACE = uuid4()


class FakeGateway:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.inserted: list[tuple[str, dict]] = []

    async def service_insert(self, path, *, payload):
        if self.error:
            raise self.error
        self.inserted.append((path, payload))


def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_embedding_cost_eur_per_million=Decimal("0.1166"),
    )


async def test_a_completion_is_recorded_with_its_tokens_and_cost() -> None:
    gateway = FakeGateway()
    run_id = uuid4()

    await UsageLedger(gateway, settings()).record(
        workspace_id=WORKSPACE,
        run_id=run_id,
        source="layer_b",
        kind="completion",
        model="gpt-6-astra",
        calls=1,
        input_tokens=32_000,
        output_tokens=4_000,
        cost_eur=Decimal("0.466"),
    )

    [(table, payload)] = gateway.inserted
    assert table == "ai_usage"
    assert payload["run_id"] == str(run_id)
    assert payload["document_id"] is None
    assert payload["total_tokens"] == 36_000
    assert payload["cost_eur"] == "0.466"


async def test_nothing_is_recorded_without_a_call() -> None:
    gateway = FakeGateway()

    await UsageLedger(gateway, settings()).record(
        workspace_id=WORKSPACE,
        source="layer_c",
        kind="completion",
        model=None,
        calls=0,
        input_tokens=0,
        output_tokens=0,
        cost_eur=Decimal("0"),
    )

    assert gateway.inserted == []


async def test_embeddings_are_recorded_even_when_the_work_fails() -> None:
    gateway = FakeGateway()
    document_id = uuid4()

    with pytest.raises(RuntimeError):
        async with UsageLedger(gateway, settings()).embeddings(
            workspace_id=WORKSPACE,
            document_id=document_id,
            source="ingestion",
            model="text-embedding-3-large",
        ):
            count_embedding(600_000)
            count_embedding(400_000)
            raise RuntimeError("indexation en échec")

    [(_, payload)] = gateway.inserted
    assert payload["kind"] == "embedding"
    assert payload["document_id"] == str(document_id)
    assert (payload["calls"], payload["input_tokens"]) == (2, 1_000_000)
    assert payload["output_tokens"] == 0
    assert payload["cost_eur"] == "0.116600"


async def test_concurrent_documents_keep_their_own_count() -> None:
    gateway = FakeGateway()
    ledger = UsageLedger(gateway, settings())

    async def document(tokens: int) -> None:
        async with ledger.embeddings(
            workspace_id=WORKSPACE, source="ingestion", model="m"
        ):
            await asyncio.sleep(0)
            await asyncio.gather(*(asyncio.sleep(0) for _ in range(2)))
            count_embedding(tokens)

    await asyncio.gather(document(100), document(250))

    assert sorted(payload["input_tokens"] for _, payload in gateway.inserted) == [
        100,
        250,
    ]


async def test_a_recording_failure_does_not_interrupt_the_work() -> None:
    gateway = FakeGateway(error=RuntimeError("Supabase indisponible"))

    async with UsageLedger(gateway, settings()).embeddings(
        workspace_id=WORKSPACE, source="classification", model="m"
    ):
        count_embedding(10)


def test_counting_outside_a_meter_is_ignored() -> None:
    count_embedding(10)
