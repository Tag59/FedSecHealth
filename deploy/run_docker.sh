#!/usr/bin/env bash
# Submit the FedSecHealth Flower App to the dockerised federation
# (start it first with `docker compose -f deploy/docker-compose.yml up -d --build`).
#
#   ROUNDS=20 STRATEGY=median bash deploy/run_docker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ROUNDS="${ROUNDS:-10}"
STRATEGY="${STRATEGY:-fedavg}"
DP="${DP:-false}"
DP_EPSILON="${DP_EPSILON:-8.0}"
export PATH="$ROOT/.venv/bin:$PATH"

export FLWR_HOME="$ROOT/deploy/.flwr_home"
mkdir -p "$FLWR_HOME" "$ROOT/results/flower_deployment_docker"
cat > "$FLWR_HOME/config.toml" <<EOF
[superlink]
default = "fedsechealth-docker"

[superlink.fedsechealth-docker]
address = "127.0.0.1:8000"
root-certificates = "$ROOT/deploy/certs/ca.crt"
EOF

flwr run "$ROOT/deploy/flower" fedsechealth-docker --stream \
  --run-config "num-hospitals=3 num-server-rounds=$ROUNDS strategy='$STRATEGY' dp-enabled=$DP dp-epsilon=$DP_EPSILON output='/app/results/flower_deployment_docker/history.json'"
echo "history in results/flower_deployment_docker/history.json"
