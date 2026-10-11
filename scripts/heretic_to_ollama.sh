#!/usr/bin/env bash
# Turn a Heretic-saved Hugging Face model (safetensors folder) into an Ollama model.
# Usage: heretic_to_ollama.sh <folder Heretic saved> [ollama-model-name]
set -euo pipefail

SRC="${1:?Usage: $0 <folder Heretic saved> [ollama-model-name]}"
NAME="${2:-berylize-heretic}"
WORK="${WORK:-$HOME/heretic-build}"
[ -f "$SRC/config.json" ] || { echo "No config.json in $SRC - is that the folder Heretic saved?"; exit 1; }
command -v ollama >/dev/null || { echo "ollama not found on this machine"; exit 1; }

mkdir -p "$WORK" && cd "$WORK"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp

# isolated venv so the converter's CPU torch cannot replace the CUDA torch used by Heretic
python3 -m venv venv-gguf
. venv-gguf/bin/activate
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# q8_0 is near-lossless and converts directly (no llama-quantize build needed)
python llama.cpp/convert_hf_to_gguf.py "$SRC" --outfile "$WORK/$NAME.gguf" --outtype q8_0
deactivate

cat > "$WORK/Modelfile" <<EOF
FROM $WORK/$NAME.gguf
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 8192
EOF

ollama create "$NAME" -f "$WORK/Modelfile"
echo; ollama list
echo; echo "Done. Reload the Crane Builder page and pick '$NAME' in the model dropdown."
