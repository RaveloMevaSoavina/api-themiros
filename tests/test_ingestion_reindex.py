from app.modules.ingestion.reindex import reindex


class FakeGateway:
    """`segment_embeddings` en mémoire : segment_id -> (modèle, texte)."""

    def __init__(self, rows: dict[str, tuple[str, str]]) -> None:
        self.rows = rows
        self.upserts: list[list[dict]] = []

    async def service_select(self, path, *, params=None):
        model = params["model"].removeprefix("neq.")
        return [
            {"segment_id": key, "workspace_id": "w1", "segments": {"text": text}}
            for key, (current, text) in sorted(self.rows.items())
            if current != model
        ][: int(params["limit"])]

    async def service_upsert(self, path, *, payload, on_conflict):
        assert on_conflict == "segment_id"
        self.upserts.append(payload)
        for row in payload:
            self.rows[row["segment_id"]] = (
                row["model"],
                self.rows[row["segment_id"]][1],
            )


class FakeEmbedder:
    model = "text-embedding-3-large"

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


async def test_only_segments_of_another_model_are_reembedded() -> None:
    gateway = FakeGateway(
        {
            "s1": ("old-model", "Analyse des besoins."),
            "s2": ("text-embedding-3-large", "Déjà à jour."),
            "s3": ("old-model", "Théorie du changement."),
        }
    )

    assert await reindex(gateway, FakeEmbedder()) == 2

    assert [row["segment_id"] for row in gateway.upserts[0]] == ["s1", "s3"]
    assert gateway.upserts[0][0] == {
        "segment_id": "s1",
        "workspace_id": "w1",
        "embedding": "[1.00000000,0.00000000]",
        "model": "text-embedding-3-large",
    }
    assert {model for model, _ in gateway.rows.values()} == {"text-embedding-3-large"}
