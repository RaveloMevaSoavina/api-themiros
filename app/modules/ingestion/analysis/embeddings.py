"""Embeddings et similarité sémantique (spec §7.3 et §9.1)."""

import asyncio
import math
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.ingestion.errors import IngestionError, embedding_unavailable

RETRY_ATTEMPTS = 3


class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def build_embedder(settings: Settings) -> "Embedder":
    if settings.ingestion_embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return LocalEmbedder(settings)


_MODELS: dict[tuple[str, str | None, int | None], Any] = {}
_MODELS_LOCK = threading.Lock()

# Modèles ONNX publiés mais absents du catalogue fastembed : même
# tokenizer, mean pooling et normalisation que le modèle d'origine.
CUSTOM_LOCAL_MODELS: dict[str, dict[str, Any]] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2-int8": {
        "hf": "xenova/paraphrase-multilingual-mpnet-base-v2",
        "model_file": "onnx/model_quantized.onnx",
        "dim": 768,
        "size_in_gb": 0.28,
    },
}
_REGISTERED: set[str] = set()


def _register_custom_model(model: str) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    spec = CUSTOM_LOCAL_MODELS[model]
    TextEmbedding.add_custom_model(
        model=model,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=spec["hf"]),
        dim=spec["dim"],
        model_file=spec["model_file"],
        size_in_gb=spec["size_in_gb"],
    )
    _REGISTERED.add(model)


def load_local_model(model: str, cache_dir: str | None, threads: int | None) -> Any:
    """Un seul chargement par processus."""
    key = (model, cache_dir, threads)
    with _MODELS_LOCK:
        if key not in _MODELS:
            from fastembed import TextEmbedding

            if model in CUSTOM_LOCAL_MODELS and model not in _REGISTERED:
                _register_custom_model(model)
            _MODELS[key] = TextEmbedding(
                model_name=model, cache_dir=cache_dir, threads=threads
            )
        return _MODELS[key]


def split_windows(text: str, max_words: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), max_words)
    ]


class LocalEmbedder:
    """Modèle multilingue exécuté sur le CPU du worker (fastembed, ONNX) :
    ni clé, ni coût par document, aucun texte envoyé à un tiers.

    `paraphrase-multilingual-mpnet-base-v2` a été entraîné sur des textes
    courts (128 tokens) alors qu'un segment en vise 512 : chaque segment est
    découpé en fenêtres de quelques dizaines de mots, et son vecteur est la
    moyenne des fenêtres pondérée par leur longueur, normalisée L2."""

    def __init__(
        self,
        settings: Settings,
        loader: Callable[[str, str | None, int | None], Any] = load_local_model,
    ) -> None:
        self.model = settings.ingestion_embedding_model
        self._dimensions = settings.ingestion_embedding_dimensions
        self._batch_size = settings.ingestion_embedding_batch_size
        self._window_words = settings.ingestion_embedding_window_words
        self._cache_dir = settings.ingestion_embedding_cache_dir
        self._threads = settings.ingestion_embedding_threads
        self._loader = loader

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts))

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            model = self._loader(self.model, self._cache_dir, self._threads)
        except Exception as error:
            # Téléchargement du modèle impossible : on réessaiera plus tard.
            raise embedding_unavailable() from error

        windows: list[str] = []
        owners: list[int] = []
        weights: list[int] = []
        for index, text in enumerate(texts):
            for window in split_windows(text, self._window_words):
                windows.append(window)
                owners.append(index)
                weights.append(max(1, len(window.split())))

        matrix = np.asarray(
            list(model.embed(windows, batch_size=self._batch_size)),
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != self._dimensions:
            raise IngestionError(
                status_code=500,
                code="EMBEDDING_DIMENSIONS_MISMATCH",
                detail=(
                    "Le modèle d'embeddings ne produit pas la dimension "
                    "attendue par l'index"
                ),
                context={
                    "model": self.model,
                    "expected": self._dimensions,
                    "actual": int(matrix.shape[-1]) if matrix.size else 0,
                },
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        weighted = matrix / norms * np.asarray(weights, dtype=np.float64)[:, None]
        pooled = np.zeros((len(texts), self._dimensions))
        np.add.at(pooled, np.asarray(owners), weighted)
        return [l2_normalize(row) for row in pooled.tolist()]


class OpenAIEmbedder:
    """`text-embedding-3-*` réduit à la dimension de l'index, normalisation
    L2, 3 tentatives avec backoff."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self.model = settings.ingestion_embedding_model
        self._dimensions = settings.ingestion_embedding_dimensions
        self._batch_size = settings.ingestion_embedding_batch_size
        self._client = client
        if client is None and settings.openai_api_key:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=0,
            )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._client is None:
            raise IngestionError(
                status_code=503,
                code="AI_PROVIDER_NOT_CONFIGURED",
                detail="OPENAI_API_KEY est requis pour l'analyse sémantique",
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                assert self._client is not None
                response = await self._client.embeddings.create(
                    model=self.model, dimensions=self._dimensions, input=batch
                )
                return [l2_normalize(item.embedding) for item in response.data]
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ) as error:
                if attempt == RETRY_ATTEMPTS - 1:
                    raise embedding_unavailable() from error
                await asyncio.sleep(2**attempt)
            except openai.APIStatusError as error:
                raise ApiError(
                    status_code=502,
                    code="EMBEDDING_REQUEST_REJECTED",
                    detail="Le service d'embeddings a refusé la requête",
                    context={"upstream_status": error.status_code},
                ) from error
        raise embedding_unavailable()


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Vecteurs déjà normalisés : la similarité cosinus est le produit scalaire."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def sample_indices(count: int, size: int) -> list[int]:
    """Indices répartis régulièrement du premier au dernier segment.
    Déterministe : un même document donne toujours le même échantillon."""
    if count <= size:
        return list(range(count))
    if size == 1:
        return [count // 2]
    return sorted({round(step * (count - 1) / (size - 1)) for step in range(size)})


def semantic_score(
    vectors: Sequence[Sequence[float]],
    fingerprint: Sequence[float],
    weights: Sequence[int],
) -> int:
    """Spec §7.3 : moyenne des similarités pondérée par la longueur des
    segments (ceux de l'échantillon, voir `sample_indices`), convertie en
    score 0-100."""
    total = sum(weights)
    if not vectors or total <= 0:
        return 0
    weighted = sum(
        cosine(vector, fingerprint) * weight
        for vector, weight in zip(vectors, weights, strict=True)
    )
    return max(0, min(100, int(weighted / total * 100)))
