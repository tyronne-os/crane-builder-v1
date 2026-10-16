#!/bin/bash
# Runs ON berylize-node the moment capacity is grabbed. Mounts the 500GB armory
# disk, then pulls the full diffusion-model manifest with the HF token for speed.
# Launched from WSL by deploy_armory.sh. Idempotent — safe to re-run.
set -u
ARMORY=/mnt/armory
LOG=$ARMORY/download.log

echo "[armory] begin $(date -u)" | tee -a /tmp/armory-boot.log

# 1. Mount the 500GB data disk (GCP names devices by disk name).
DEV=/dev/disk/by-id/google-berylize-extra-storage
mkdir -p "$ARMORY"
if [ -e "$DEV" ]; then
  if ! grep -q " $ARMORY " /etc/fstab 2>/dev/null; then
    blkid "$DEV" >/dev/null 2>&1 || mkfs.ext4 -F "$DEV"   # format only if blank
    echo "$DEV $ARMORY ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
  mount "$ARMORY" 2>/dev/null || mount -a
  echo "[armory] disk mounted at $ARMORY: $(df -h $ARMORY | tail -1)"
else
  echo "[armory] WARNING: data disk not found at $DEV — writing to boot disk fallback"
fi

# 2. Tools
command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq aria2 >/dev/null 2>&1; }

# 3. HF auth (token arrives via stdin from the launcher — kept off the scripts)
read -r HFTOK </dev/stdin
[ -n "$HFTOK" ] && echo "Authorization: Bearer $HFTOK" > /root/.hfhdr && chmod 600 /root/.hfhdr

# 4. The manifest — the full 10-model research armory (9 real; MiniMax H3 has no
#    released weights — noted in the log, not a download).
mkdir -p "$ARMORY"
cat > "$ARMORY/urls.txt" <<'EOF'
https://huggingface.co/yabaimimi/edwixx__diffusiongemma-26B-A4B-it-HERETIC-Uncensored-GGUF/resolve/main/edwixx__diffusiongemma-26B-A4B-it-HERETIC-Uncensored-Q4_K_M.gguf
	out=diffusiongemma-26B-heretic-Q4_K_M.gguf
https://huggingface.co/QuantStack/Wan2.2-S2V-14B-GGUF/resolve/main/Wan2.2-S2V-14B-Q4_K_M.gguf
	out=wan2.2-s2v-14b-Q4_K_M.gguf
https://huggingface.co/vantagewithai/LTX2.3-10Eros-GGUF/resolve/main/10Eros_v1-Q4_K_M.gguf
	out=ltx2.3-10eros-Q4_K_M.gguf
https://huggingface.co/jayn7/HunyuanVideo-1.5_T2V_480p-GGUF/resolve/main/480p/hunyuanvideo1.5_480p_t2v-Q5_K_M.gguf
	out=hunyuanvideo1.5-480p-t2v-Q5_K_M.gguf
https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf
	out=qwen-image-2.1-turbo-uncens-AD-Q4_K.gguf
https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q4_K_M.gguf
	out=wan2.2-ti2v-5b-Q4_K_M.gguf
https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors
	out=wan2.2_vae.safetensors
https://huggingface.co/OlegSkutte/sdxl-turbo-GGUF/resolve/main/sd_xl_turbo_1.0.q8_0.gguf
	out=sdxl-turbo-q8_0.gguf
https://huggingface.co/Motif-Technologies/Motif-Video-2B-GGUF/resolve/main/motifv-2b-dev-Q4_K_M.gguf
	out=motif-video-2b-Q4_K_M.gguf
EOF
echo "[armory] MiniMax H3: no released weights — excluded from downloads" >> "$LOG"

# 5. Fire the downloads — 8 connections per file, 2 files at a time, resumable.
cd "$ARMORY"
nohup aria2c -i urls.txt -x8 -j2 -c --header="Authorization: Bearer $HFTOK" \
  --auto-file-renaming=false --allow-overwrite=true \
  --summary-interval=60 >> "$LOG" 2>&1 &
echo "[armory] downloads launched into $ARMORY ($(du -sh urls.txt | cut -f1) manifest, 65.4GB expected)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
echo "[armory] boot sequence complete $(date -u)"
