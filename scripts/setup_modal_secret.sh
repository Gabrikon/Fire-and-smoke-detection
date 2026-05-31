#!/usr/bin/env bash
# Create/refresh the Modal secret bundle from your local .env.
# Usage: bash scripts/setup_modal_secret.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and fill it in first."
  exit 1
fi

modal secret create fire-vlm-secrets --from-dotenv .env --force
echo "Modal secret 'fire-vlm-secrets' created/updated from .env."
