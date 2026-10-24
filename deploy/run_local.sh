#!/usr/bin/env bash
# Local TLS deployment: one SuperLink (server) and N SuperNodes (hospitals) as
# separate processes on this machine, then submit the FedSecHealth Flower App.
#
#   bash deploy/run_local.sh                      # 3 hospitals, 10 rounds, FedAvg
#   N_HOSPITALS=5 ROUNDS=20 STRATEGY=median bash deploy/run_local.sh
#   DP=true DP_EPSILON=8 bash deploy/run_local.sh # DP-SGD in every hospital
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
N="${N_HOSPITALS:-3}"
ROUNDS="${ROUNDS:-10}"
STRATEGY="${STRATEGY:-fedavg}"
DP="${DP:-false}"
DP_EPSILON="${DP_EPSILON:-8.0}"
OUT="$ROOT/results/flower_deployment"
CERTS="$ROOT/deploy/certs"
export PATH="$ROOT/.venv/bin:$PATH" PYTHONWARNINGS=ignore

bash "$CERTS/generate.sh"
mkdir -p "$OUT"

# Project-local Flower CLI configuration (leaves ~/.flwr untouched).
export FLWR_HOME="$ROOT/deploy/.flwr_home"
mkdir -p "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" <<EOF
[superlink]
default = "fedsechealth-local"

[superlink.fedsechealth-local]
address = "127.0.0.1:8000"
root-certificates = "$CERTS/ca.crt"
EOF

pids=()
cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "starting SuperLink (TLS) ..."
# No runtime dependency installation: apps run in this pinned, audited environment
# instead of whatever a submitted app's pyproject would pull from PyPI.
flower-superlink \
  --disable-runtime-dependency-installation \
  --ssl-ca-certfile "$CERTS/ca.crt" \
  --ssl-certfile "$CERTS/server.crt" \
  --ssl-keyfile "$CERTS/server.key" \
  > "$OUT/superlink.log" 2>&1 &
pids+=($!)
sleep 4

for i in $(seq 0 $((N - 1))); do
  echo "starting hospital $i ..."
  flower-supernode \
    --superlink 127.0.0.1:9092 \
    --root-certificates "$CERTS/ca.crt" \
    --node-config "partition-id=$i num-partitions=$N" \
    --port $((9094 + i)) \
    > "$OUT/hospital-$i.log" 2>&1 &
  pids+=($!)
done
sleep 4

flwr run "$ROOT/deploy/flower" fedsechealth-local --stream \
  --run-config "num-hospitals=$N num-server-rounds=$ROUNDS strategy='$STRATEGY' dp-enabled=$DP dp-epsilon=$DP_EPSILON output='$OUT/history.json'"

echo "logs and history in $OUT"
