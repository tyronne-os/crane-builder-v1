#!/bin/bash
# Mirror of berylize-node's `startup-script` instance metadata.
#
# This is the source of truth for what the GPU box does on boot. It is not run on
# the VM directly — it is applied with:
#
#   gcloud compute instances add-metadata berylize-node --zone=us-east1-c \
#     --metadata-from-file=startup-script=scripts/startup_vm.sh
#
# Metadata is what survives a stop, a preemption or host maintenance. Anything
# started by hand dies with the instance, which is why the registry is here and
# not just launched once over SSH.
#
# Deliberately no `set -e`: a failure in the voice section must not stop the
# registry from coming up, and vice versa. The composer falls back to Ollama's
# /v1/models when the registry is absent, so a failure here costs detail, not chat.
export HOME=/root
log() { echo "[crane] $(date -u +%H:%M:%S) $*"; }
log "startup begin"

M=/opt/berylize/small-models
mkdir -p "$M/voice"

if command -v ollama >/dev/null; then
  systemctl start ollama
  for i in $(seq 1 60); do curl -sf localhost:11434/api/tags >/dev/null && break; sleep 2; done
  for m in hf.co/mradermacher/Qwen2.5-14B-Instruct-Heretic-GGUF:Q4_K_M hf.co/mradermacher/Qwen2.5-Coder-1.5B-Instruct-abliterated-i1-GGUF:Q4_K_M; do
    ollama list | grep -q "${m%%:*}" || { log "pulling $m"; ollama pull "$m"; }
  done
fi

[ -x /opt/hv/bin/python ] || python3 -m venv /opt/hv
/opt/hv/bin/pip install -q -U huggingface_hub >/dev/null 2>&1

/opt/hv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
base = "/opt/berylize/small-models/voice"
snapshot_download("hexgrad/Kokoro-82M", local_dir=f"{base}/kokoro-82m",
    allow_patterns=["*.json", "*.pth", "*.md", "voices/af_*", "voices/bf_*"])
print("[crane] kokoro-82m ready", flush=True)
snapshot_download("kyutai/tts-1.6b-en_fr", local_dir=f"{base}/kyutai-tts-1.6b")
snapshot_download("kyutai/tts-voices", local_dir=f"{base}/kyutai-tts-1.6b/voices",
    allow_patterns=["expresso/ex04-*channel1*", "expresso/ex03-ex02_*channel2*", "expresso/ex01-ex02_*channel2*", "README.md"])
print("[crane] kyutai-tts-1.6b ready", flush=True)
PY

log "voice models:"; du -sh "$M"/voice/* 2>/dev/null
log "llms:"; ollama list 2>/dev/null
# Crane model registry (added 2026-10-14): read-only. Reports which models are
# actually installed, so the composer's STORED IN GPU section is never a guess.
# Absent or failing, the composer falls back to Ollama /v1/models, so this cannot
# break chat.
if [ -f /opt/berylize/model_registry.py ]; then
  pkill -f model_registry.py 2>/dev/null
  nohup python3 /opt/berylize/model_registry.py >/tmp/registry.log 2>&1 &
  sleep 2
  log "model registry: $(curl -sf --max-time 10 localhost:11436/health || echo FAILED)"
else
  log "model_registry.py missing - STORED IN GPU falls back to ollama /v1/models"
fi

log "startup done"
