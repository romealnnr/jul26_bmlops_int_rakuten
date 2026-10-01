#!/usr/bin/env bash
# Environnement de développement reproductible.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[setup] .env créé depuis .env.example"
fi

mkdir -p models reports src/data/raw

if command -v uv >/dev/null 2>&1; then
  echo "[setup] uv détecté"
  uv venv --python 3.12
  uv pip install -e ".[dev]"
else
  echo "[setup] uv absent, fallback venv + pip"
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/pip install -e ".[dev]"
fi

echo
echo "[setup] Terminé. Étapes suivantes :"
echo "  source .venv/bin/activate"
echo "  docker compose up -d db"
echo "  python -m src.data.make_dataset"
echo "  python -m src.models.training"
echo "  uvicorn src.api.main:app --reload"
