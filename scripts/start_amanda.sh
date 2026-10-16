#!/bin/bash
# Amanda — Co-founder model (Qwen2.5-Coder-1.5B-Instruct-abliterated, Q4_K_M)
# Served on CPU at port 11437; the CORS proxy maps browser port 11438 to her.
# The Crane Builder composer auto-discovers her as "Amanda-Qwen-1.5B" under CPU.
SRV=~/llama.cpp/build/bin/llama-server
MODEL="/home/trell/THE STAFF/uncensored/Qwen2.5-Coder-1.5B-Instruct-abliterated.i1-Q4_K_M.gguf"
PROXY=/home/trell/crane-builder-v1/scripts/cors_proxy.py

# Stop a previous Amanda instance, keep the 8080 STAFF server alone.
pkill -f "llama-server.*11437" 2>/dev/null; sleep 1

# Benchmarked on the Ryzen 3 7320U: auto (8) threads ~7.9 tok/s beats 4 threads
# ~6.9 tok/s, so let llama-server auto-detect — no --threads flag.
nohup "$SRV" -m "$MODEL" --alias Amanda-Qwen-1.5B --port 11437 --ctx-size 8192 > /tmp/amanda-11437.log 2>&1 &

# The proxy gives the hosted builder page CORS + private-network access.
if ! curl -m 2 -s -o /dev/null http://localhost:11438/; then
  nohup python3 "$PROXY" > /tmp/cors_proxy.log 2>&1 &
fi

sleep 20
curl -m 10 -s http://localhost:11437/health && echo " :11437 Amanda-Qwen-1.5B UP"
curl -m 10 -s http://localhost:11438/v1/models | head -c 200 && echo " :11438 proxy -> Amanda"
