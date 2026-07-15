#!/usr/bin/env bash
# Build (and optionally push) the deps-only image for linux/amd64.
#   DOCKERHUB_USER=<you> bash docker/build_push.sh              # build only (--load)
#   DOCKERHUB_USER=<you> bash docker/build_push.sh --push       # build + push to Docker Hub
# Tag encodes the dep stack; bump -vN on every uv.lock change (immutable tags —
# vast hosts cache layers by tag).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DOCKERHUB_USER:?export DOCKERHUB_USER=<docker hub namespace>}"
TAG="${TAG:-cu128-t271-v1}"
IMAGE="docker.io/${DOCKERHUB_USER}/querent-deps:${TAG}"

MODE="--load"
if [ "${1:-}" = "--push" ]; then MODE="--push"; fi

echo "building ${IMAGE} (${MODE})"
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "${IMAGE}" ${MODE} .
echo "done: ${IMAGE}"
