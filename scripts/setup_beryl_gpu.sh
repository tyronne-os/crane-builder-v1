#!/usr/bin/env bash
# Bring Beryl to life on your GPU machine (berylize-node: Ubuntu 22.04, NVIDIA driver 580).
#
#   bash setup_beryl_gpu.sh all      everything: ears + voice (worker), brain (Ollama + Qwen3.5), face (FlashHead)
#   bash setup_beryl_gpu.sh          worker only (Qwen3-ASR ears + Qwen3-TTS voice)
#   bash setup_beryl_gpu.sh brain    worker + brain
#   bash setup_beryl_gpu.sh face     worker + face
#   bash setup_beryl_gpu.sh bench    worker + Phase 0 speed test of the face model
#
# Each part is a service that starts at boot and loads its models at start, so there is no lag once they report ready.
# On a GPU under 30 GB (your L4 has 24) the lean profile is chosen automatically: 0.6B ears, full 1.7B voice, Qwen3.5-4B
# brain, FlashHead Lite face, about 17 GB together. Override with PROFILE=full.
#
# One tunnel from your PC carries listening, voice, brain and face (the worker relays the brain and the face):
#   gcloud compute ssh berylize-node --zone us-east1-c --project posh-eden -- -N -L 8882:localhost:8882
# then in the builder set both "Beryl GPU server" and the model endpoint to http://127.0.0.1:8882/v1
set -euo pipefail
MODE=${1:-worker}
DIR=${BERYL_HOME:-/opt/beryl}
RAW=https://raw.githubusercontent.com/tyronne-os/crane-builder-v1/main
mkdir -p "$DIR" && cd "$DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee gpu.txt
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PROFILE=${PROFILE:-$([ "$VRAM_MB" -lt 30000 ] && echo lean || echo full)}
echo "GPU memory ${VRAM_MB} MB -> profile $PROFILE"

for f in beryl_gpu_server.py beryl_turns.py beryl_face.py beryl_director.py personaplex_bridge.py beryl_face_server.py; do
  curl -fsSL "$RAW/scripts/$f" -o "$f"
done
curl -fsSL "$RAW/frontend/uploads/beryl.jpeg" -o beryl.jpeg

have_systemd() { command -v systemctl >/dev/null && [ -d /run/systemd/system ]; }
service() {  # name, workdir, command, extra Environment= lines
  local name=$1 wd=$2 cmd=$3 env=${4:-}
  if have_systemd; then
    printf '[Unit]\nDescription=%s\nAfter=network-online.target\n\n[Service]\nWorkingDirectory=%s\n%bExecStart=%s\nRestart=always\nRestartSec=5\n\n[Install]\nWantedBy=multi-user.target\n' \
      "$name" "$wd" "$env" "$cmd" > "/etc/systemd/system/$name.service"
    systemctl daemon-reload && systemctl enable "$name.service" && systemctl restart "$name.service"
    echo "$name installed as a boot service. Log: journalctl -u $name -f"
  else
    (cd "$wd" && nohup bash -c "$cmd" > "$DIR/$name.log" 2>&1 &)
    echo "$name started in the background. Log: $DIR/$name.log"
  fi
}

# --- worker: ears + voice -------------------------------------------------------------------------------------
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# qwen-asr pins transformers==4.57.6 and qwen-tts pins 4.57.3; both run on 4.57.6, so qwen-tts goes in without its pin.
.venv/bin/pip install -q qwen-asr fastapi "uvicorn[standard]" soundfile numpy httpx pillow
.venv/bin/pip install -q --no-deps qwen-tts && .venv/bin/pip install -q onnxruntime einops sox
.venv/bin/pip install -q wheel setuptools && .venv/bin/pip install -q flash-attn --no-build-isolation || echo "flash-attn skipped (optional, saves memory)"
service beryl "$DIR" "$DIR/.venv/bin/python $DIR/beryl_gpu_server.py" \
  "Environment=BERYL_PROFILE=$PROFILE\nEnvironment=HF_HOME=$DIR/hf\n"

# --- brain: Ollama + Qwen3.5 ----------------------------------------------------------------------------------
if [ "$MODE" = brain ] || [ "$MODE" = all ]; then
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  BRAIN=${OLLAMA_MODEL:-$([ "$PROFILE" = lean ] && echo hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M || echo hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M)}
  ollama pull "$BRAIN"
  echo "Brain ready: $BRAIN (pick it from the model menu in the builder)"
fi

# --- face: SoulX-FlashHead Lite, its own venv (it pins torch 2.7.1 and transformers 4.57.3) --------------------------
if [ "$MODE" = face ] || [ "$MODE" = all ] || [ "$MODE" = bench ]; then
  command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
  [ -d SoulX-FlashHead ] || git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead
  cd SoulX-FlashHead
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  # requirements.txt pins nvidia-nccl-cu12==2.27.3, which conflicts with torch 2.7.1's own nccl; that pin is for multi-GPU only.
  grep -v -i 'nvidia-nccl' requirements.txt > requirements.beryl.txt
  .venv/bin/pip install -q -r requirements.beryl.txt "huggingface_hub[cli]" fastapi "uvicorn[standard]" pillow
  .venv/bin/pip install -q ninja && MAX_JOBS=4 .venv/bin/pip install -q flash_attn==2.8.0.post2 --no-build-isolation
  .venv/bin/huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/SoulX-FlashHead-1_3B
  .venv/bin/huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h
  cd "$DIR"
  if [ "$MODE" != bench ]; then
    service beryl-face "$DIR/SoulX-FlashHead" "$DIR/SoulX-FlashHead/.venv/bin/python $DIR/beryl_face_server.py" \
      "Environment=PYTHONPATH=$DIR/SoulX-FlashHead:$DIR\nEnvironment=FLASHHEAD_DIR=$DIR/SoulX-FlashHead\nEnvironment=BERYL_PORTRAIT=$DIR/beryl.jpeg\nEnvironment=FLASHHEAD_MODEL=lite\n"
  fi
fi

# --- Phase 0 bench: can the face keep up in real time on this GPU? ------------------------------------------------
if [ "$MODE" = bench ]; then
  cd "$DIR/SoulX-FlashHead"
  for i in $(seq 1 120); do curl -sf localhost:8882/health | grep -q '"voice_ready": *true' && break; sleep 5; done
  curl -sf localhost:8882/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input":"Hey, you made it. So I was thinking about the landing page, and honestly, I love where it is going. Want to walk through it together?"}' -o beryl_line.wav
  ffmpeg -loglevel error -y -i beryl_line.wav -ar 16000 -ac 1 beryl_16k.wav
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 beryl_16k.wav)
  start=$(date +%s.%N)
  .venv/bin/python generate_video.py --ckpt_dir models/SoulX-FlashHead-1_3B --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite --cond_image "$DIR/beryl.jpeg" --audio_path beryl_16k.wav --audio_encode_mode stream > gen.log 2>&1
  end=$(date +%s.%N)
  echo "audio ${dur}s rendered in $(python3 -c "print(round($end - $start, 2))")s (real time needs the second number below the first)" | tee -a "$DIR/bench.txt"
fi

echo
echo "Check readiness:  curl -s localhost:8882/health   and   curl -s localhost:8883/health"
