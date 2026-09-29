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

## Tests and quality

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

## Production deployment

The production stack runs the API and an automatic HTTPS reverse proxy with
Docker Compose. See [DEPLOYMENT.md](DEPLOYMENT.md) for server prerequisites,
first deployment, updates and operational commands.

## Structure

```text
app/
  api/             HTTP routing and authentication dependencies
  core/            configuration, errors and middleware
  infrastructure/  external services such as Supabase
  modules/         business modules (frameworks, ingestion, runs, ...)
```
