#!/usr/bin/env bash
# Prepare and start the dockerised federation (server + 3 hospitals).
#   bash deploy/docker_up.sh && bash deploy/run_docker.sh
#   docker compose -f deploy/docker-compose.yml down
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash deploy/certs/generate.sh
# Created by us (not by Docker as root) so the non-root containers can write to it.
mkdir -p results/flower_deployment_docker
export MEDMNIST_ROOT="${MEDMNIST_ROOT:-$HOME/.medmnist}"
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps
