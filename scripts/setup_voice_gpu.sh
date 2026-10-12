#!/usr/bin/env bash
# Run on the GPU VM (berylize-node). Installs Kokoro + VibeVoice-Realtime-0.5B and starts Crane Voice on 127.0.0.1:8880.
# From your PC, tunnel it alongside Ollama (local 8881, since your PC's own Kokoro server uses 8880):
#   gcloud compute ssh berylize-node --zone us-east1-c --project posh-eden -- -N -L 11434:localhost:11434 -L 8881:localhost:8880
set -euo pipefail
DIR=/opt/berylize/voice
sudo mkdir -p "$DIR" && sudo chown "$USER" "$DIR" && cd "$DIR"
curl -fsSL https://raw.githubusercontent.com/tyronne-os/crane-builder-v1/main/scripts/voice_server.py -o voice_server.py
[ -d VibeVoice ] || git clone --depth 1 https://github.com/microsoft/VibeVoice
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e "VibeVoice[streamingtts]" kokoro soundfile fastapi uvicorn
sudo apt-get install -y -qq espeak-ng >/dev/null || true
nohup .venv/bin/python voice_server.py > voice.log 2>&1 &
echo "Crane Voice starting (first request downloads the model). Log: $DIR/voice.log"
