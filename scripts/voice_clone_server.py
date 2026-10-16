#!/usr/bin/env python3
"""Amanda Voice Studio — clone server on :8091.

Zero-shot voice cloning with Coqui XTTS-v2. The Crane Builder Voice Studio
records/uploads a 10–30 s reference, POSTs it here, and speaks with the cloned
voice forever after. Voice IDs are `clone:<name>`.

Endpoints:
  GET  /health                    liveness
  GET  /v1/voices                 known clones
  POST /clone                     {name, audio_b64} -> registers a clone
  POST /v1/audio/speech           {voice:"clone:<name>", input, speed} -> audio/wav
  POST /audio/speech              same (OpenAI-style path the app posts to)

First run downloads the XTTS-v2 model (~1.8 GB) into ./xtts-data.
CPU synthesis runs around real-time; a GPU build of torch makes it instant.

Setup (one-time, Python 3.11 — the coqui engine needs it):
  uv venv clone-env --python 3.11
  uv pip install -p clone-env/bin/python coqui-tts "transformers==4.57.1" flask flask-cors soundfile numpy
  uv pip install -p clone-env/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  bash ~/staff-build/start-amanda-stack.sh --clone
"""
import base64
import io
import os
import time
import wave

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice-clones")
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts-data")
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

_tts = None
_model = None


def tts():
    global _tts, _model
    if _tts is None:
        from TTS.api import TTS
        print("loading XTTS-v2 (first run downloads ~1.8 GB)…", flush=True)
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        _model = _tts.synthesizer.tts_model
    return _tts, _model


def clone_path(name):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip().lower()) or "voice"
    return os.path.join(DATA_DIR, safe + ".wav")


def synth_to_wav(text, speaker_wav, speed):
    t, model = tts()
    wav = t.tts(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        speed=max(0.5, min(2.0, speed or 1.0)),
    )
    buf = io.BytesIO()
    sf.write(buf, np.array(wav), 24000, format="WAV")
    return buf.getvalue()


@app.route("/health")
@app.route("/v1/voices")
def health():
    clones = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".wav"))
    return jsonify({"status": "ok", "voices": ["clone:" + c for c in clones], "gpu": bool(torch.cuda.is_available())})


@app.route("/clone", methods=["POST"])
def clone():
    data = request.get_json(force=True, silent=True) or {}
    name = str(data.get("name", "")).strip()
    b64 = str(data.get("audio_b64", ""))
    if not name or not b64:
        return jsonify({"error": "name and audio_b64 required"}), 400
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return jsonify({"error": "bad base64"}), 400
    # Anything the browser records (webm/ogg) is decoded by soundfile via ffmpeg-free
    # path; XTTS wants 24 kHz mono wav, so we ask the model to condition on it and
    # resample through librosa-free soundfile round-trip. ffmpeg in PATH helps for webm.
    tmp = os.path.join(DATA_DIR, name.replace("/", "_") + ".src")
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        audio, sr = sf.read(tmp)
    except Exception:
        os.system(f"ffmpeg -y -i {tmp} -ar 24000 -ac 1 {tmp}.wav >/dev/null 2>&1")
        if not os.path.exists(tmp + ".wav"):
            return jsonify({"error": "could not decode reference — install ffmpeg for webm/ogg"}), 400
        audio, sr = sf.read(tmp + ".wav")
    if hasattr(audio, "shape") and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        import subprocess
        with sf.SoundFile(tmp + ".ref.wav", "w", samplerate=sr, channels=1) as f:
            f.write(np.asarray(audio).astype("float32"))
        os.system(f"ffmpeg -y -i {tmp}.ref.wav -ar 24000 -ac 1 {tmp}.ref24.wav >/dev/null 2>&1")
        if os.path.exists(tmp + ".ref24.wav"):
            audio, _ = sf.read(tmp + ".ref24.wav")
        else:
            return jsonify({"error": "ffmpeg needed to resample reference to 24 kHz"}), 400
    out = clone_path(name)
    with sf.SoundFile(out, "w", samplerate=24000, channels=1) as f:
        f.write(np.asarray(audio).astype("float32"))
    for junk in (tmp, tmp + ".wav", tmp + ".ref.wav", tmp + ".ref24.wav"):
        if os.path.exists(junk):
            os.remove(junk)
    return jsonify({"voice": "clone:" + name, "stored": out})


@app.route("/v1/audio/speech", methods=["POST"])
@app.route("/audio/speech", methods=["POST"])
def speech():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text") or data.get("input") or "")[:2000]
    voice = str(data.get("voice", ""))
    speed = float(data.get("speed", 1.0))
    if not text.strip():
        return jsonify({"error": "text required"}), 400
    name = voice[6:] if voice.startswith("clone:") else voice
    path = clone_path(name)
    if not os.path.exists(path):
        return jsonify({"error": f"clone '{name}' not found — record it first"}), 404
    wav = synth_to_wav(text, path, speed)
    return Response(wav, mimetype="audio/wav")


if __name__ == "__main__":
    print("Amanda Voice Studio clone server on :8091 — record a reference in the app", flush=True)
    app.run("127.0.0.1", 8091, threaded=False)
