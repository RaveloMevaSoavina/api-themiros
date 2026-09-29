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

echo "Building the API image..."
"${compose[@]}" build --pull api

echo "Starting the API..."
"${compose[@]}" up -d --remove-orphans api

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
    "${compose[@]}" ps
    echo "Deployment complete."
    exit 0
  fi
  sleep 2
done

echo "The API did not become healthy in time." >&2
"${compose[@]}" logs --tail=100 api >&2
exit 1
