#!/bin/bash
# Start a llama.cpp OpenAI-compatible server for one GGUF model.
#
# Replaces Ollama. Two roles, chosen by the --device flag:
#   gpu  CUDA, for the 14B.  Listens on 11434 (the port Crane already talks to).
#   cpu  Pure CPU, for sub-1B models. Listens on 11437.
#
# Why this instead of Ollama: llama-server is a single static binary with no
# daemon, no database and no blob store. It reads the GGUF sitting on disk.
# Models are files you can ls, move, back up and delete - which is what makes
# "what does my GPU actually have" answerable without a guessing UI.
#
# Usage:
#   llama_server.sh gpu  <path/to.gguf>
#   llama_server.sh cpu  <path/to.gguf>
set -euo pipefail

ROLE="${1:?Usage: $0 gpu|cpu <model.gguf>}"
GGUF="${2:?Usage: $0 gpu|cpu <model.gguf>}"

[ -f "$GGUF" ] || { echo "[-] No such model file: $GGUF"; exit 1; }

BIN="${LLAMA_SERVER_BIN:-/opt/berylize/llama.cpp/build/bin/llama-server}"
[ -x "$BIN" ] || { echo "[-] llama-server not built at $BIN - run scripts/setup_llama_cpp.sh"; exit 1; }

MODELS_DIR="${CRANE_MODELS_DIR:-/opt/berylize/models}"
LOG="/tmp/llama-server-$ROLE.log"

case "$ROLE" in
  gpu)
    PORT=11434
    # Leave VRAM headroom: the FlashHead face service (~5.8GB) and the Beryl voice
    # worker (~6.6GB) are resident on the same L4, leaving roughly 10GB. A 14B at
    # Q4_K_M is 9GB, so context is trimmed to keep a usable margin.
    EXTRA=(--n-gpu-layers 99 --ctx-size 8192 --parallel 1)
    ;;
  cpu)
    PORT=11437
    THREADS="${CRANE_CPU_THREADS:-$(nproc)}"
    # No GPU layers: this endpoint exists so small models run without touching VRAM,
    # leaving the whole card free for the 14B.
    EXTRA=(--n-gpu-layers 0 --threads "$THREADS" --ctx-size 4096)
    ;;
  *) echo "[-] Unknown role '$ROLE' (expected gpu or cpu)"; exit 1 ;;
esac

# Derive a stable model id from the filename so the composer shows something
# readable: qwen3-1b-instruct-q4_k_m.gguf -> qwen3-1b-instruct-q4_k_m
MODEL_ID="${MODEL_ID:-$(basename "$GGUF" .gguf)}"

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 1

echo "[*] $MODEL_ID ($ROLE) -> 127.0.0.1:$PORT  threads=${CRANE_CPU_THREADS:-auto}"
echo "[*] log: $LOG"

nohup "$BIN" \
  --model "$GGUF" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --alias "$MODEL_ID" \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1 &

for i in $(seq 1 90); do
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[+] up after ${i}s: $(curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models")"
    exit 0
  fi
  if ! kill -0 $! 2>/dev/null; then
    echo "[-] llama-server exited early. Last lines of $LOG:"; tail -20 "$LOG"; exit 1
  fi
  sleep 1
done

echo "[-] Timed out waiting for :$PORT. Last lines of $LOG:"; tail -20 "$LOG"; exit 1
