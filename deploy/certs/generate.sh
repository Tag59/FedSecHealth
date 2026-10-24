#!/usr/bin/env bash
# Create a private certificate authority and a TLS certificate for the SuperLink.
#
# Hospitals (SuperNodes) and the `flwr` CLI trust ca.crt and verify that the
# server presents a certificate signed by it for the expected name
# (localhost, 127.0.0.1 or "superlink" inside Docker).
# For a demo only: in production the CA key would live in an HSM / a real PKI.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f ca.crt && -f server.crt && "${1:-}" != "--force" ]]; then
  echo "certificates already exist in $(pwd) (use --force to regenerate)"
  exit 0
fi

openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -sha256 -days 365 -nodes \
  -keyout ca.key -out ca.crt -subj "/O=FedSecHealth/CN=FedSecHealth demo CA" 2>/dev/null

openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
  -keyout server.key -out server.csr -subj "/O=FedSecHealth/CN=superlink" 2>/dev/null

cat > server.ext <<'EOF'
subjectAltName = DNS:localhost, DNS:superlink, IP:127.0.0.1
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 365 -sha256 -extfile server.ext 2>/dev/null
rm -f server.csr server.ext ca.srl
chmod 600 ca.key server.key
chmod 644 ca.crt server.crt
openssl verify -CAfile ca.crt server.crt
