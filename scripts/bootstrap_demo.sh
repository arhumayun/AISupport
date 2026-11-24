#!/usr/bin/env bash
set -euo pipefail

pull_model=false
if [[ "${1:-}" == "--pull-model" ]]; then
  pull_model=true
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--pull-model]" >&2
  exit 2
fi

if [[ ! -f .env ]]; then
  echo "Create .env from .env.example and replace every placeholder first." >&2
  exit 2
fi
required=(
  AI_SUPPORT_SESSION_SECRET
  AI_SUPPORT_DEMO_LOGIN_CODE
  AI_SUPPORT_POSTGRES_PASSWORD
  AI_SUPPORT_MINIO_USER
  AI_SUPPORT_MINIO_PASSWORD
  AI_SUPPORT_GRAFANA_USER
  AI_SUPPORT_GRAFANA_PASSWORD
)
for name in "${required[@]}"; do
  if ! grep -q "^${name}=" .env; then
    echo "Missing required setting ${name} in .env." >&2
    exit 2
  fi
done
if grep -Eq '=(change-me|replace-)' .env; then
  echo "Replace every change-me/replace-* placeholder in .env." >&2
  exit 2
fi

model="$(sed -n 's/^AI_SUPPORT_OLLAMA_MODEL=//p' .env | tail -n 1)"
model="${model:-llama3.1:latest}"
export AI_SUPPORT_GIT_COMMIT
AI_SUPPORT_GIT_COMMIT="$(git rev-parse HEAD)"
export AI_SUPPORT_GIT_DIRTY=false
if [[ -n "$(git status --short)" ]]; then
  AI_SUPPORT_GIT_DIRTY=true
fi

docker compose config --quiet
docker compose build api minio
docker compose up -d postgres minio mailpit mlflow ollama

if [[ "$pull_model" == true ]]; then
  docker compose exec -T ollama ollama pull "$model"
elif ! docker compose exec -T ollama ollama show "$model" >/dev/null 2>&1; then
  echo "Ollama model '$model' is absent. Re-run with --pull-model to explicitly download it." >&2
  exit 2
fi

docker compose run --rm --no-deps api ai-support bootstrap-demo --rows 600
docker compose up -d api streamlit prometheus grafana
docker compose run --rm --no-deps api \
  python -m ai_support.deployment.verify --base-url http://api:8000

echo "Bootstrap and authenticated acceptance verification completed successfully."
