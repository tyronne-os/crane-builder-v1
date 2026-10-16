#!/bin/bash
# Local builder — serves the composer from your own machine so the page talks
# to your local Amandas localhost→localhost (no Chrome local-network permission
# walls). The hosted Space stays for remote use.
set -u
cd /home/trell/crane-builder-v1/frontend
if curl -m 2 -s -o /dev/null http://localhost:8080/ 2>/dev/null; then
  echo "✓ Local builder already running"
else
  nohup python3 -m http.server 8080 --bind 127.0.0.1 > /tmp/builder-local.log 2>&1 &
  sleep 1
  echo "→ started on :8080"
fi
echo ""
echo "  OPEN:  http://localhost:8080/app.html"
echo "  This copy talks to Amanda, Mini, voice, and memory with zero permission prompts."
echo "  Note: keys/settings live per-site — paste your HF token + keys once here."
