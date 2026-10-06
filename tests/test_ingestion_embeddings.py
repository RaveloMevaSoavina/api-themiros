import math

import pytest

from app.core.config import Settings
from app.modules.ingestion.analysis.embeddings import (
    LocalEmbedder,
    OpenAIEmbedder,
    build_embedder,
    sample_indices,
    split_windows,
)
from app.modules.ingestion.errors import is_retryable


class FakeModel:
    """Vecteur 3D selon le premier mot de la fenêtre, non normalisé."""

    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.windows: list[str] = []

    def embed(self, windows, batch_size):
        self.windows.extend(windows)
        for window in windows:
            first = window.split()[0] if window.split() else ""
            vector = [0.0] * self.dimensions
            vector[0 if first == "eau" else 1] = 5.0
            yield vector


def settings() -> Settings:
    # Fenêtres de 4 mots pour un test lisible (sous le minimum de production).
    return Settings().model_copy(
        update={
            "ingestion_embedding_dimensions": 3,
            "ingestion_embedding_window_words": 4,
        }
    )


def test_split_windows_keeps_every_word():
    text = "un deux trois quatre cinq six sept huit neuf"
    assert split_windows(text, 4) == [
        "un deux trois quatre",
        "cinq six sept huit",
        "neuf",
    ]
    assert split_windows("   ", 4) == ["   "]


async def test_local_embedder_pools_windows_by_length():
    model = FakeModel()
    embedder = LocalEmbedder(settings(), loader=lambda *_: model)

    [short, long] = await embedder.embed(["eau irrigation", "eau a b c autre d e f g"])

    assert short == pytest.approx([1.0, 0.0, 0.0])
    # Deux fenêtres de 4 mots (eau / autre) et une d'un mot (g → autre).
    expected = [4.0, 5.0, 0.0]
    norm = math.sqrt(sum(value * value for value in expected))
    assert long == pytest.approx([value / norm for value in expected])
    assert len(model.windows) == 4


async def test_local_embedder_rejects_wrong_dimensions():
    embedder = LocalEmbedder(settings(), loader=lambda *_: FakeModel(5))
    with pytest.raises(Exception) as error:
        await embedder.embed(["texte"])
    assert getattr(error.value, "code", None) == "EMBEDDING_DIMENSIONS_MISMATCH"
    assert not is_retryable(error.value)


async def test_local_model_download_failure_is_retried():
    def broken(*_):
        raise OSError("network down")

    embedder = LocalEmbedder(settings(), loader=broken)
    with pytest.raises(Exception) as error:
        await embedder.embed(["texte"])
    assert getattr(error.value, "code", None) == "EMBEDDING_SERVICE_UNAVAILABLE"
    assert is_retryable(error.value)


async def test_local_embedder_skips_empty_input():
    def never(*_):
        raise AssertionError("le modèle ne doit pas être chargé")

    assert await LocalEmbedder(settings(), loader=never).embed([]) == []


def test_provider_selects_embedder():
    assert isinstance(build_embedder(Settings()), LocalEmbedder)
    assert isinstance(
        build_embedder(Settings(ingestion_embedding_provider="openai")),
        OpenAIEmbedder,
    )


def test_sample_indices_spread_over_the_document():
    assert sample_indices(10, 4) == [0, 3, 6, 9]
    assert sample_indices(3, 32) == [0, 1, 2]
    assert sample_indices(5, 1) == [2]
    assert sample_indices(0, 32) == []
