# Themiros API

FastAPI backend for deterministic evaluation rules, framework generation and
the document-analysis pipeline.

## Start locally

```bash
cp .env.example .env
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn app.main:app --reload
```

Useful endpoints:

- `GET /health`: liveness probe
- `GET /ready`: Supabase readiness probe
- `POST /api/v1/frameworks/generate-pillars`: authenticated pillar-generation
  contract
- `GET /docs`: OpenAPI UI in development

Set `AI_PROVIDER=openai` and `OPENAI_API_KEY` to enable pillar generation. The
provider uses the Responses API with Structured Outputs, then applies domain
validation to references, applicable criteria, pillar count and weights.

## Document ingestion (E4)

Uploads are registered by the dashboard through the Supabase function
`register_document`, which queues an `ingestion_jobs` row. The worker consumes
that queue: extraction and OCR, language and country detection, relevance
scoring against the workspace fingerprint, decision, segmentation and indexing.

```bash
# Tesseract with fra, eng, por and spa language data must be installed.
.venv/bin/pip install -e '.[dev,nlp]'
.venv/bin/python -m spacy download xx_ent_wiki_sm   # optional NER model
.venv/bin/python -m app.modules.ingestion.worker
```

The worker needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. It computes
embeddings locally with fastembed (`paraphrase-multilingual-mpnet-base-v2`,
int8 ONNX, 768 dimensions, CPU). The model is baked into the Docker image under
`/opt/fastembed`; locally it is downloaded once to
`INGESTION_EMBEDDING_CACHE_DIR`. Set `INGESTION_EMBEDDING_PROVIDER=openai` with
`OPENAI_API_KEY` and `INGESTION_EMBEDDING_MODEL=text-embedding-3-small` to use
OpenAI instead; changing the model requires reindexing. The relevance score
embeds an evenly spread sample of segments (`INGESTION_RELEVANCE_SAMPLE_SEGMENTS`,
default 32); the remaining segments are embedded at indexing time, only for
compliant or human-approved documents.

Business thresholds
(70/40, bonuses and penalties, off-topic keywords, size limits) live in the
versioned `ingestion_settings` table: rows without `workspace_id` hold the
platform defaults, and workspace admins add their own versions from
Settings › Document screening (`update_ingestion_settings`). The worker uses
the latest version of the document's workspace.

Endpoints from the API catalogue (§6):

- `POST /api/v1/ingestion/extract`: extraction and OCR of an uploaded file
- `POST /api/v1/ingestion/qualify`: relevance score and status
- `POST /api/v1/ingestion/segment`: segmentation and indexing
- `POST /api/v1/ingestion/full`: complete asynchronous ingestion (202)
- `GET /api/v1/jobs/{id}`: ingestion job status

## Tests and quality

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

## Production deployment

The production stack runs the API with Docker Compose on a host-only port. The
server's existing Caddy instance provides HTTPS. See
[DEPLOYMENT.md](DEPLOYMENT.md) for the first deployment, Caddy configuration,
updates and operational commands.

## Structure

```text
app/
  api/             HTTP routing and authentication dependencies
  core/            configuration, errors and middleware
  infrastructure/  external services such as Supabase
  modules/         business modules (frameworks, ingestion, runs, ...)
```
