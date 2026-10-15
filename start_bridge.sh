#!/bin/bash
# Self-activating SSH tunnel and CORS proxy bridge for Crane GPU node

echo "[*] Starting local CORS proxy..."
cd ~/crane-builder-v1
python3 scripts/cors_proxy.py &
PROXY_PID=$!

echo "[*] Establishing SSH port forwarding to berylize-node (gcloud compute)..."
gcloud compute ssh berylize-node --zone=us-east1-c -- -N -L 11434:localhost:11434 -L 11436:localhost:11436 &
TUNNEL_PID=$!

echo "[+] Bridge active! Proxy PID: $PROXY_PID | Tunnel PID: $TUNNEL_PID"
echo "[*] 11434 = Ollama (chat)   11436 = model registry (what is actually installed)"
echo "[*] Keep this terminal open. Press Ctrl+C to stop both services."

trap "kill $PROXY_PID $TUNNEL_PID; exit" INT TERM
wait
