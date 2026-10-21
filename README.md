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

The worker needs `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` and
`OPENAI_API_KEY`. It computes embeddings with OpenAI `text-embedding-3-large`
reduced to 768 dimensions. `INGESTION_EMBEDDING_PROVIDER=local` switches back
to fastembed (`paraphrase-multilingual-mpnet-base-v2`, int8 ONNX, CPU, baked
into the Docker image under `/opt/fastembed`). Changing the model requires
reindexing the corpus (`python -m app.modules.ingestion.reindex`, before the
workers restart) and recalibrating the similarity thresholds. The relevance score
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

## Analysis runs (E6, layers A and B)

The dashboard launches a run with the Supabase function `create_run`, which
freezes the corpus and queues the run. The worker process also consumes that
queue (`app/modules/runs`): it reads the frozen inputs (`run_inputs`), computes
intermediate variables, corpus confidence and layer A scores in pure Python
(`domain.py`), computes layer B (below), then writes everything in one
transaction (`complete_run`). Layer C is not computed yet: its step is marked
`skipped`.

Layer B (`layer_b.py`, `grader.py`) rates each retained criterion 0-3. Each
active sub-question is embedded with the segment model and matched against the
run's frozen corpus (`run_sub_question_matches`, top `RUN_LAYER_B_TOP_K`
above `RUN_LAYER_B_MIN_SIMILARITY`, default 0.40). Criteria with fewer than 50%
of their sub-questions matched are `non_conclu` without any AI call. The others
are rated by the `grade_sub_questions` prompt in a single AI call per run
(split only beyond `RUN_LAYER_B_MAX_INPUT_TOKENS`), with deduplicated,
truncated excerpts (`RUN_LAYER_B_EXCERPTS_PER_QUESTION`,
`RUN_LAYER_B_EXCERPT_CHARS`). A rating counts only if it cites one of the
question's own excerpts. When AI is disabled or fails, no rating is produced:
criteria stay `non_conclu` and the run still completes. AI usage (model,
tokens, cost, errors) is stored in `rules_applied.engine.layer_b.ai`.

Variable scoring is deterministic: 1 (`present`) when strong detections
(similarity ≥ `RUN_STRONG_SIMILARITY`, default 0.55) come from at least the
workspace's number of independent sources, 0.5 (`partiel`) for any other
detection, `non_renseigne` otherwise. The engine never concludes absence (0).
The thresholds and `ENGINE_VERSION` are stored in each run's `rules_applied`.
A run whose corpus classification is still pending is deferred
(`RUN_DEFER_SECONDS`) for at most `RUN_CLASSIFICATION_WAIT_SECONDS`.

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
