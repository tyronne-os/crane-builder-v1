#!/bin/bash
# Build llama.cpp and fetch the GGUFs that Crane serves. Run once on berylize-node.
#
# This replaces the Ollama install. GGUFs are downloaded straight from Hugging Face
# into /opt/berylize/models as ordinary files, so the set of models on the box is
# just `ls /opt/berylize/models` - no database, no daemon, no guessing.
#
# Safe to re-run: the llama.cpp build is skipped if already present, and each model
# is skipped if its file already exists.
set -euo pipefail

PREFIX="${CRANE_PREFIX:-/opt/berylize}"
MODELS_DIR="${CRANE_MODELS_DIR:-$PREFIX/models}"
SRC="$PREFIX/llama.cpp"
JOBS="${JOBS:-$(nproc)}"

# ggml.org hosts the CUDA build; the CPU build is the default target.
WITH_CUDA="${WITH_CUDA:-1}"

log() { echo "[setup] $*"; }

mkdir -p "$MODELS_DIR"

# ---- llama.cpp -------------------------------------------------------------
if [ ! -x "$SRC/build/bin/llama-server" ]; then
  [ -d "$SRC" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
  log "building llama.cpp (CUDA=$WITH_CUDA, this takes a few minutes)"
  if [ "$WITH_CUDA" = "1" ]; then
    (cd "$SRC" && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release >/dev/null \
                 && cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli >/dev/null)
  else
    (cd "$SRC" && cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null \
                 && cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli >/dev/null)
  fi
  log "built $SRC/build/bin/llama-server"
else
  log "llama-server already built, skipping"
fi

# ---- models ----------------------------------------------------------------
# name<TAB>repo<TAB>file-in-repo<TAB>role
# `role` decides which endpoint serves it: gpu for the 14B, cpu for sub-1B.
# All three are refusal-suppressed (abliterated / heretic), so the assistant does
# not refuse on-topic work. Filenames and sizes were verified against the repos.
MODELS=(
"Qwen2.5-14B-Heretic-q4_k_m	mradermacher/Qwen2.5-14B-Instruct-Heretic-GGUF	Qwen2.5-14B-Instruct-Heretic.Q4_K_M.gguf	gpu"
"Qwen2.5-Coder-1.5B-abliterated-q4_k_m	mradermacher/Qwen2.5-Coder-1.5B-Instruct-abliterated-i1-GGUF	Qwen2.5-Coder-1.5B-Instruct-abliterated.i1-Q4_K_M.gguf	cpu"
"Qwen2.5-0.5B-heretic-q4_k_m	mradermacher/Heretic-Qwen2.5-0.5b-RBase-GGUF	Heretic-Qwen2.5-0.5b-RBase.Q4_K_M.gguf	cpu"
)

log "downloading models into $MODELS_DIR"
for row in "${MODELS[@]}"; do
  IFS=$'\t' read -r name repo file role <<<"$row"
  out="$MODELS_DIR/$name.gguf"
  if [ -f "$out" ]; then
    log "have $name ($(du -h "$out" | cut -f1))"
    continue
  fi
  log "downloading $name from $repo/$file"
  curl -fL --retry 3 --retry-delay 3 -o "$out.part" \
    "https://huggingface.co/$repo/resolve/main/$file?download=true"
  mv "$out.part" "$out"
done

# ---- summary ---------------------------------------------------------------
echo
echo "Models on disk in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (none)"
echo
echo "Start them with:"
for row in "${MODELS[@]}"; do
  IFS=$'\t' read -r name repo file role <<<"$row"
  echo "  scripts/llama_server.sh $role $MODELS_DIR/$name.gguf"
done
echo
echo "GPU endpoint : 127.0.0.1:11434  (Crane's default gpuBase)"
echo "CPU endpoint : 127.0.0.1:11437"
