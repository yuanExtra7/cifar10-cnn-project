#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] apt update + install docker.io (CLI + dockerd)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends docker.io ca-certificates iptables

echo
echo "[2/5] show versions"
docker --version || true
dockerd --version || true

echo
echo "[3/5] start dockerd (no systemd in this environment)"
mkdir -p /var/run /var/lib/docker

# Try overlay2 first; if it fails, fall back to vfs (slower but often works in restricted containers).
set +e
dockerd \
  --host=unix:///var/run/docker.sock \
  --storage-driver=overlay2 \
  --iptables=false \
  --ip-masq=false \
  --bridge=none \
  >/var/log/dockerd.log 2>&1 &
DOCKERD_PID=$!
set -e

sleep 3

if ! docker info >/dev/null 2>&1; then
  echo "[warn] overlay2 start failed; falling back to vfs (check /var/log/dockerd.log)"
  kill "${DOCKERD_PID}" || true
  sleep 1
  dockerd \
    --host=unix:///var/run/docker.sock \
    --storage-driver=vfs \
    --iptables=false \
    --ip-masq=false \
    --bridge=none \
    >/var/log/dockerd.log 2>&1 &
  DOCKERD_PID=$!
  sleep 3
fi

echo
echo "[4/5] verify docker is up"
docker info | sed -n '1,80p'

echo
echo "[5/5] done"
echo "dockerd pid: ${DOCKERD_PID}"
echo "dockerd log: /var/log/dockerd.log"
echo "NOTE: if docker still fails, the platform may block dockerd in this container."

