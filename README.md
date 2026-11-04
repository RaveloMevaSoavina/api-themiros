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

## Analysis runs (E6, layers A, B and C, restitution)

The dashboard launches a run with the Supabase function `create_run`, which
freezes the corpus and queues the run. The worker process also consumes that
queue (`app/modules/runs`): it reads the frozen inputs (`run_inputs`), computes
intermediate variables, corpus confidence and layer A scores in pure Python
(`domain.py`), computes layer B, layer C and the restitution (below), then
writes everything in one transaction (`complete_run`).

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

Layer C (`layer_c.py`) compares layer A scores with layer B ratings using
deterministic rules (ALERT-01 to 06), then has every gap qualified in a single
`detect_incoherence` call. Without AI, gaps are published with their rule's
default severity and message (`qualified = false`).

The restitution (`restitution.py`) selects up to three progress points,
weaknesses and areas to verify, plus the action origins, with deterministic
rules; weaknesses are relative (the weakest pillars below 80/100 and criteria
below 2.4/3). A single `formulate_restitution` call (prompt v1.2) writes the
findings, a cautious narrative conclusion (two or three paragraphs) and each
action worded for the three personas, in one or two full sentences with
their reason. The same call also translates the findings, the
conclusion and the actions into the other interface languages (fr, en),
stored in `translations` columns; the dashboard and exports fall back to the
run's language when a translation is missing. Without
AI, findings keep a factual label, with no conclusion and no action plan; the
synthesis `status` says why (`not_needed`, `disabled` or `failed`, otherwise
`completed`). AI usage is stored in `rules_applied.engine.restitution.ai`.

Variable scoring is deterministic: 1 (`present`) when strong detections
(similarity ≥ `RUN_STRONG_SIMILARITY`, default 0.55) come from at least the
workspace's number of independent sources, 0.5 (`partiel`) for any other
detection, `non_renseigne` otherwise. The engine never concludes absence (0).
The thresholds and `ENGINE_VERSION` are stored in each run's `rules_applied`.
A run whose corpus classification is still pending is deferred
(`RUN_DEFER_SECONDS`) for at most `RUN_CLASSIFICATION_WAIT_SECONDS`.

## AI usage

Every OpenAI call is recorded in `ai_usage` (migration
`20261028000002_ai_usage.sql`) with the tokens billed, as returned by the API,
and its cost: one row per document (`ingestion`), validated framework
(`classification`), run step (`layer_b_search`, `layer_b`, `layer_c`,
`restitution`) and framework generation (`pillar_generation`, written by a
database trigger). `app/modules/usage/ledger.py` holds the ledger; embeddings
are counted by `OpenAIEmbedder` in a per-task meter, so concurrent documents
keep separate counts. Failed work is still recorded, since successful calls
are billed, and a recording failure never fails the work.

Costs use `OPENAI_INPUT_COST_EUR_PER_MILLION`,
`OPENAI_OUTPUT_COST_EUR_PER_MILLION` and
`OPENAI_EMBEDDING_COST_EUR_PER_MILLION` at call time (0 when unset). Cached
input tokens are priced as regular input, so costs are slightly overstated
when OpenAI's prompt cache applies. The dashboard reads the totals through
`workspace_ai_usage` (account menu › AI usage, for the open workspace).

## Run exports (E14)

`POST /api/v1/exports/{run_id}/{pdf|docx|xlsx|csv}` (body:
`{"locale": "fr"|"en", "persona": "decideur"|"pmu"|"analyste"}`) returns the
report of a completed run as a file (`Content-Disposition` gives its name).
Everything is read with the caller's token, so RLS decides what they can
export. `app/modules/exports/report.py` builds one report shared by every
format, in the order of the screens: corpus limits first (always, including
unconcluded criteria, undocumented variables, missing layers, changed corpus
and missing validation), then comparison, pillars (layer A), criteria
(layer B), alerts (layer C), synthesis and the action plan for the requested
persona. Evidence becomes numbered notes, one per cited document page,
quoting at most the two excerpts with the highest similarity and stating how
many were left out; consecutive note numbers are shown as ranges
(`[131–140]`). An exploratory run carries a warning banner on the first page.

Unlike the API catalogue (§9), the file is generated during the request
rather than queued and polled, so there is no `exports` table or storage.
Each export is recorded in `audit_events` (`report_exported`, or
`report_export_failed`) with its format, run, author and timestamp; an
export that cannot be recorded fails.

## Demandes de démo (landing page)

`POST /api/v1/demo-requests` reçoit le formulaire de
`landing-page/request-demo.html`. Route **publique**, sans jeton.

1. Valide les champs (`name`, `organisation`, `email` obligatoires ;
   `context`, `source` facultatifs) : 422 sinon.
2. Honeypot `website` rempli : 201 sans rien stocker ni envoyer.
3. Au plus `DEMO_REQUEST_RATE_LIMIT` demandes enregistrées (3) par adresse
   IP sur `DEMO_REQUEST_RATE_WINDOW_SECONDS` (600), vérifié en base : 429
   au-delà. L'adresse est la dernière entrée de `X-Forwarded-For`, posée par
   Caddy.
4. Enregistre la demande dans `public.demo_requests` (clé serveur).
5. Prévient l'équipe par e-mail via Resend (`Reply-To` : le demandeur), puis
   passe `notified` à vrai.

### Anti-robots

En plus de la limite en base, un garde en mémoire
(`app/modules/demo_requests/guard.py`) filtre la route avant même la lecture
du corps :

| Barrière            | Règle par défaut                                                           | Réponse                          |
| ------------------- | -------------------------------------------------------------------------- | -------------------------------- |
| Tentatives par IP   | Toute requête compte (valide, invalide, honeypot) : plus de `DEMO_REQUEST_ATTEMPT_LIMIT` (10) en `DEMO_REQUEST_ATTEMPT_WINDOW_SECONDS` (600) bannit l'adresse `DEMO_REQUEST_BAN_SECONDS` (3 600). | 429 `DEMO_REQUEST_BLOCKED`       |
| Honeypot            | L'adresse est bannie d'office ; la requête piégée reçoit quand même 201.  | 429 aux suivantes                |
| Plafond global      | Au plus `DEMO_REQUEST_GLOBAL_LIMIT` (30) demandes enregistrées par heure, toutes adresses confondues (robot réparti sur plusieurs IP). | 429 `DEMO_REQUEST_GLOBAL_LIMIT`  |
| Taille du corps     | 16 Kio, dans Caddy (`deploy/Caddyfile.api`) et dans l'API.                | 413                              |

En IPv6, la clé est le préfixe /64, qu'un abonné peut faire tourner à
volonté. Les compteurs vivent dans le processus (uvicorn en a un seul) : un
redémarrage les remet à zéro, la limite en base reste. Les bannissements
sont journalisés (`Demo requests: … blocked`). Le garde est installé avant le
CORS pour que ses 429 portent les en-têtes CORS : le formulaire affiche
alors « patientez quelques minutes » et non l'échec générique.

Si l'e-mail échoue ou n'est pas configuré, la demande reste enregistrée et
la réponse est un succès : le backoffice (`/demo-requests`) la marque
« Non notifiée ».

Réglages (`.env.production`) :

| Variable                  | Rôle                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `CORS_ALLOWED_ORIGINS`    | Doit contenir l'origine de la landing (`https://themiros.com`, `https://www.themiros.com`). |
| `RESEND_API_KEY`          | Clé Resend. Vide : aucun e-mail.                           |
| `DEMO_REQUEST_MAIL_TO`    | Destinataires, séparés par des virgules.                   |
| `DEMO_REQUEST_MAIL_FROM`  | Expéditeur, sur un domaine vérifié chez Resend.           |
| `BACKOFFICE_URL`          | Facultatif : lien vers l'écran de suivi dans l'e-mail.    |

Prérequis : la table `demo_requests`
(`landing-page/supabase/migrations/20260918000001_demo_requests.sql`) et,
pour le suivi, la migration
`dashboard-themiros/supabase/migrations/20261104000001_backoffice_demo_requests.sql`.

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
