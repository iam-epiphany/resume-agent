#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
  echo "[FAIL] Docker is not running." >&2
  exit 1
fi

export RESUME_APP_IMAGE="${RESUME_APP_IMAGE:-resumemind/app:latest}"

build_args=""
if [ "${1:-}" = "--build" ] || [ "${1:-}" = "build" ] || ! docker image inspect "$RESUME_APP_IMAGE" >/dev/null 2>&1; then
  build_args="--build"
fi

# shellcheck disable=SC2086
docker compose up -d $build_args

app_port="${RESUME_APP_PORT:-8000}"
qdrant_port="${RESUME_QDRANT_HTTP_PORT:-6333}"
echo "ResumeMind is starting at http://127.0.0.1:$app_port"
echo "Qdrant dashboard: http://127.0.0.1:$qdrant_port/dashboard"
echo "Logs: docker compose logs -f app"
