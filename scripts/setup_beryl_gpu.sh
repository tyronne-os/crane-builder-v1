#!/usr/bin/env bash
# Run on your GPU machine (berylize-node on Google Cloud, or any Ubuntu box with an NVIDIA driver).
#   bash setup_beryl_gpu.sh            install the worker (Qwen3-ASR ears + Qwen3-TTS voice) as a service that starts at boot
#   PROFILE=lean bash setup_beryl_gpu.sh   same, with the 0.6B ears and voice so everything fits a 24 GB card (L4)
#   bash setup_beryl_gpu.sh brain      also install Ollama with the Qwen3.5 brain (OLLAMA_MODEL to override)
#   bash setup_beryl_gpu.sh bench      also run the Phase 0 SoulX-FlashHead Lite speed test on Beryl's photo
# The service loads every model at boot, so there is no start-up lag once /health says ready.
# One tunnel from your PC carries listening, voice and the brain (the worker passes the brain through to Ollama):
#   gcloud compute ssh berylize-node --zone us-east1-c --project posh-eden -- -N -L 8882:localhost:8882
# then in the builder set both "Beryl GPU server" and the model endpoint to http://127.0.0.1:8882/v1
set -euo pipefail
DIR=${BERYL_HOME:-/opt/beryl}
RAW=https://raw.githubusercontent.com/tyronne-os/crane-builder-v1/main
mkdir -p "$DIR" && cd "$DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee gpu.txt

curl -fsSL "$RAW/scripts/beryl_gpu_server.py" -o beryl_gpu_server.py
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# qwen-asr pins transformers==4.57.6 and qwen-tts pins 4.57.3; both run on 4.57.6, so install qwen-tts without its pin.
.venv/bin/pip install -q qwen-asr fastapi "uvicorn[standard]" soundfile numpy
.venv/bin/pip install -q --no-deps qwen-tts && .venv/bin/pip install -q onnxruntime einops sox
.venv/bin/pip install -q flash-attn --no-build-isolation || echo "flash-attn skipped (optional, saves memory)"
pkill -f beryl_gpu_server.py || true
PROFILE=${PROFILE:-full}
if command -v systemctl >/dev/null && [ -d /run/systemd/system ]; then
  cat > /etc/systemd/system/beryl.service <<UNIT
[Unit]
Description=Beryl worker (Qwen3-ASR + Qwen3-TTS)
After=network-online.target

[Service]
WorkingDirectory=$DIR
Environment=BERYL_PROFILE=$PROFILE
Environment=HF_HOME=$DIR/hf
ExecStart=$DIR/.venv/bin/python $DIR/beryl_gpu_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload && systemctl enable --now beryl.service
  echo "Installed as the beryl service (starts at boot). Log: journalctl -u beryl -f"
else
  BERYL_PROFILE=$PROFILE HF_HOME="$DIR/hf" nohup .venv/bin/python beryl_gpu_server.py > beryl.log 2>&1 &
  echo "No systemd here; started in the background. Log: $DIR/beryl.log"
fi
echo "Models download once, then load at every start. Wait for: curl -s localhost:8882/health  ->  \"ready\": true"

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
