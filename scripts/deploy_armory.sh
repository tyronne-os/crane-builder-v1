#!/bin/bash
# Fired from WSL the moment the capacity grabber reports berylize-node RUNNING.
# Streams the HF token to the VM over SSH (never written to any script file),
# then the remote script mounts the armory disk and launches all downloads.
set -euo pipefail
ZONE=us-east1-c
VM=berylize-node

HFTOK=$(grep huggingface.co ~/.git-credentials | head -1 | sed 's|.*://[^:]*:\([^@]*\)@.*|\1|')
[ -n "$HFTOK" ] || { echo "no HF token found"; exit 1; }

echo "[deploy] waiting for SSH to come up on $VM…"
for i in $(seq 1 30); do
  if gcloud compute ssh $VM --zone=$ZONE --command='echo SSH_OK' 2>/dev/null | grep -q SSH_OK; then break; fi
  sleep 10
done

echo "[deploy] streaming the armory payload (token via stdin, not on disk)…"
gcloud compute ssh $VM --zone=$ZONE --command='bash -s' \
  < <(echo "$HFTOK"; cat "$(dirname "$0")/armory_remote.sh")

echo "[deploy] checking download progress in 60s…"
sleep 60
gcloud compute ssh $VM --zone=$ZONE --command='tail -5 /mnt/armory/download.log 2>/dev/null; df -h /mnt/armory 2>/dev/null | tail -1'
