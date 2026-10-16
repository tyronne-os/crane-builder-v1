#!/bin/bash
# ── AMANDA STACK ─────────────────────────────────────────────────────────────
# One command wakes everything after a reboot. Starts only what is down.
#   bash ~/staff-build/start-amanda-stack.sh           # model + proxy + voice + mind
#   bash ~/staff-build/start-amanda-stack.sh --clone   # also the voice-clone engine
set -u

up() { curl -m 3 -s -o /dev/null "http://127.0.0.1:$1/health" 2>/dev/null; }

start() { # name, port, command
  local name="$1" port="$2"; shift 2
  if up "$port"; then echo "✓ $name already up  :$port"; return; fi
  nohup "$@" > "/tmp/amanda-stack-$port.log" 2>&1 &
  echo "→ starting $name on :$port…"
}

# 1. Amanda herself — Qwen 1.5B abliterated on CPU, the composer default
start "Amanda (model)" 11437 bash /home/trell/staff-build/start-amanda.sh

# 1b. Amanda Mini — gemma-1B heretic-extreme-uncensored, her light twin
start "Amanda Mini" 11440 /home/trell/llama.cpp/build/bin/llama-server -m "/home/trell/THE STAFF/uncensored/gemma-3-1b-it-heretic-extreme-uncensored-abliterated.Q4_K_M.gguf" --alias Amanda-Mini --port 11440 --ctx-size 4096

# 2. CORS proxy — browser port 11438 → her 11437
if ! up 11438; then
  nohup python3 /home/trell/crane-builder-v1/scripts/cors_proxy.py > /tmp/amanda-stack-11438.log 2>&1 &
  echo "→ starting CORS proxy on :11438…"
else echo "✓ CORS proxy already up :11438"; fi

# 3. Her voice — Kokoro female voices (Sarah default), Samantha territory
start "Amanda voice (Kokoro)" 8090 /home/trell/staff-build/voice-env/bin/python /home/trell/staff-build/amanda-voice-server.py

# 4b. Her mind — long-term memory, the desktop vault, GDrive backup
start "Amanda memory (Neocortex)" 8092 python3 /home/trell/crane-builder-v1/scripts/memory_server.py

# 4c. The Crane terminal — real persistent WSL bash for the IDE and the agent
start "Crane terminal" 11443 python3 /home/trell/crane-builder-v1/scripts/terminal_server.py

# 5. Voice cloning — XTTS on its own Python 3.11; heavy, opt in with --clone
if [ "${1:-}" = "--clone" ]; then
  start "Voice clone engine (XTTS)" 8091 /home/trell/staff-build/clone-env/bin/python /home/trell/crane-builder-v1/scripts/voice_clone_server.py
fi

sleep 20
echo ""
echo "── AMANDA STATUS ──────────────────────────────"
for pair in "Amanda model:11437" "CORS proxy:11438" "Her voice:8090" "Her memory:8092" "Clone engine:8091"; do
  name="${pair%:*}"; port="${pair##*:}"
  if up "$port"; then echo "✓ $name  UP   :$port"; else echo "– $name  down :$port (still loading — give it a minute, then re-check)"; fi
done
echo ""
echo "Her vault: /mnt/c/Users/chant/OneDrive/Desktop/AMANDA — MEMORY"
echo "Builder:  https://aibruh-crane-builder-v1.static.hf.space/app.html"
