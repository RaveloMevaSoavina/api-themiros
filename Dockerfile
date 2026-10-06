FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OCR des documents scannés (spec §4.4) : français, anglais, portugais, espagnol.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
      tesseract-ocr-por tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Modèle d'embeddings embarqué dans l'image : le worker ne télécharge rien
# au démarrage.
ENV INGESTION_EMBEDDING_CACHE_DIR=/opt/fastembed

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install ".[nlp]" \
    && python -m spacy download xx_ent_wiki_sm \
    && python -c "from app.core.config import Settings; from app.modules.ingestion.analysis.embeddings import load_local_model; s = Settings(); load_local_model(s.ingestion_embedding_model, '/opt/fastembed', None)" \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app \
    && chown -R app:app /app /opt/fastembed

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

