#!/usr/bin/env bash
# Run on the GPU machine (rented RTX 5090/4090 on RunPod or Vast, or berylize-node). Ubuntu + NVIDIA driver assumed.
#   bash setup_beryl_gpu.sh            install Qwen3-ASR + Qwen3-TTS server and start it on 127.0.0.1:8882
#   bash setup_beryl_gpu.sh bench      also run the Phase 0 SoulX-FlashHead Lite speed test on Beryl's photo
#   bash setup_beryl_gpu.sh brain      also install Ollama with the Qwen3.5 brain (OLLAMA_MODEL to override)
# From your PC, tunnel the ports (replace host/port with your machine's SSH details):
#   ssh -N -L 8882:localhost:8882 -L 11434:localhost:11434 root@HOST -p PORT
set -euo pipefail
DIR=${BERYL_HOME:-/opt/beryl}
RAW=https://raw.githubusercontent.com/tyronne-os/crane-builder-v1/main
mkdir -p "$DIR" && cd "$DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee gpu.txt

curl -fsSL "$RAW/scripts/beryl_gpu_server.py" -o beryl_gpu_server.py
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q qwen-asr qwen-tts fastapi "uvicorn[standard]" soundfile numpy
.venv/bin/pip install -q flash-attn --no-build-isolation || echo "flash-attn skipped (optional, saves memory)"
pkill -f beryl_gpu_server.py || true
nohup .venv/bin/python beryl_gpu_server.py > beryl.log 2>&1 &
echo "Beryl server starting on 127.0.0.1:8882 (first request downloads the models and designs her voice). Log: $DIR/beryl.log"

if [ "${1:-}" = "brain" ]; then
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  ollama pull "${OLLAMA_MODEL:-qwen3.5:9b}"
fi

if [ "${1:-}" = "bench" ]; then
  # Phase 0: can SoulX-FlashHead Lite animate Beryl in real time on this GPU? Own venv: it pins transformers==4.57.3.
  command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
  [ -d SoulX-FlashHead ] || git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead
  cd SoulX-FlashHead
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/pip install -q -r requirements.txt "huggingface_hub[cli]"
  .venv/bin/pip install -q ninja && .venv/bin/pip install -q flash_attn==2.8.0.post2 --no-build-isolation
  .venv/bin/huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/SoulX-FlashHead-1_3B
  .venv/bin/huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h
  curl -fsSL "$RAW/frontend/uploads/beryl.jpeg" -o beryl.jpeg
  # 10 s of Beryl's real voice from the server, resampled to 16 kHz for the face model.
  for i in $(seq 1 60); do curl -sf localhost:8882/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8882/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input":"Hey, you made it. So I was thinking about the landing page, and honestly, I love where it is going. Want to walk through it together?"}' -o beryl_line.wav
  ffmpeg -loglevel error -y -i beryl_line.wav -ar 16000 -ac 1 beryl_16k.wav
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 beryl_16k.wav)
  run() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 500 > "vram_$1.txt" & mon=$!
    start=$(date +%s.%N)
    .venv/bin/python generate_video.py --ckpt_dir models/SoulX-FlashHead-1_3B --wav2vec_dir models/wav2vec2-base-960h \
      --model_type lite --cond_image beryl.jpeg --audio_path beryl_16k.wav --audio_encode_mode stream > "gen_$1.log" 2>&1
    end=$(date +%s.%N); kill $mon
    echo "$1: audio ${dur}s, generated in $(python3 -c "print(round($end - $start, 2))")s, peak VRAM $(sort -n "vram_$1.txt" | tail -1)"
  }
  run warmup >/dev/null
  run face_only | tee -a ../bench.txt
  # Full load: speech models busy at the same time, as in a real conversation.
  ( for i in 1 2 3; do curl -s localhost:8882/v1/audio/speech -H 'Content-Type: application/json' -d '{"input":"Okay, give me one second, I am thinking."}' -o /dev/null; done ) &
  run full_load | tee -a ../bench.txt
  echo "Real time needs generated-seconds <= audio-seconds (FlashHead outputs 25 fps). Results: $DIR/bench.txt, video in $DIR/SoulX-FlashHead"
fi
