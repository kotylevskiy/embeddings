#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${EMBEDDINGS_SERVICE_URL:-http://localhost:${EMBEDDINGS_PORT:-11445}}"
MODEL_ID="${EMBEDDINGS_MODEL_ID:-jinaai/jina-embeddings-v2-base-en}"

curl -sS "$SERVICE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "$MODEL_ID",
  "input": [
    "First text for embeddings",
    "Second text for embeddings",
    "Third text for embeddings"
  ]
}
JSON

echo
