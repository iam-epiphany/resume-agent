#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

docker rm -f resumemind-dev-frontend resumemind-dev-app resumemind-dev-qdrant >/dev/null 2>&1 || true
docker compose down
rm -f .run-state/docker-dev.json .run-state/runtime.json

echo "ResumeMind Docker development containers and Compose services have been stopped."
