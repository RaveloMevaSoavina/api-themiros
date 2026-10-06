#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -f .env.production ]]; then
  echo "Missing .env.production. Copy .env.production.example and fill it first." >&2
  exit 1
fi

compose=(docker compose --env-file .env.production)

echo "Validating Docker Compose configuration..."
"${compose[@]}" config --quiet

# L'API et le worker d'ingestion partagent la même image.
services=(api ingestion-worker)

echo "Building the API image..."
"${compose[@]}" build --pull "${services[@]}"

echo "Starting the API and the ingestion worker..."
"${compose[@]}" up -d --remove-orphans "${services[@]}"

echo "Waiting for the API health check..."
for attempt in {1..30}; do
  container_id="$("${compose[@]}" ps -q api 2>/dev/null || true)"
  health=""
  if [[ -n "$container_id" ]]; then
    health="$(
      docker inspect \
        --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' \
        "$container_id" 2>/dev/null || true
    )"
  fi
  if [[ "$health" == "healthy" ]]; then
    worker_id="$("${compose[@]}" ps -q ingestion-worker 2>/dev/null || true)"
    worker_state=""
    if [[ -n "$worker_id" ]]; then
      worker_state="$(docker inspect --format '{{.State.Status}}' "$worker_id" 2>/dev/null || true)"
    fi
    "${compose[@]}" ps
    if [[ "$worker_state" != "running" ]]; then
      echo "The ingestion worker is not running." >&2
      "${compose[@]}" logs --tail=100 ingestion-worker >&2
      exit 1
    fi
    echo "Deployment complete."
    exit 0
  fi
  sleep 2
done

echo "The API did not become healthy in time." >&2
"${compose[@]}" logs --tail=100 api >&2
exit 1
