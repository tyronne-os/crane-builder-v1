#!/usr/bin/env python3
"""STAFF Amanda voice server: Kokoro-ONNX female voices only, on :8090.

Two endpoints, same synth core:
  POST /v1/speak        {text, voice, speed}          -> audio/wav
  POST /v1/audio/speech  {model, voice, input, speed} -> audio/wav   (OpenAI-ish, what the app's VoiceOut posts)
No telemetry, no logging, CORS open for the hosted builder page.
"""
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# Female voices only — mature, Samantha-from-Her territory.
VOICES = {
    "sarah": "af_sarah", "bella": "af_bella", "nicole": "af_nicole", "sky": "af_sky",
    "heart": "af_heart", "aoede": "af_aoede", "kore": "af_kore", "nova": "af_nova",
    "jessica": "af_jessica", "river": "af_river", "alloy": "af_alloy",
    "emma": "bf_emma", "isabella": "bf_isabella", "alice": "bf_alice", "lily": "bf_lily",
}
MODEL = "/home/trell/staff-models/voices/kokoro-v1.0.onnx"
VOICES_BIN = "/home/trell/staff-models/voices/voices.bin"
_koko = None

def koko():
    global _koko
    if _koko is None:
        from kokoro_onnx import Kokoro
        _koko = Kokoro(MODEL, VOICES_BIN)
    return _koko

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/v1/voices"):
            body = json.dumps({"status": "ok", "voices": sorted(VOICES), "default": "sarah"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        if self.path not in ("/v1/speak", "/v1/audio/speech"):
            self.send_response(404)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            data = {}
        # Both shapes accepted: {text, voice, speed} and {input, voice, speed}.
        text = str(data.get("text") or data.get("input") or "")[:2000]
        voice = VOICES.get(str(data.get("voice", "sarah")).lower(), "af_sarah")
        speed = float(data.get("speed", 1.0))
        if not text.strip():
            self.send_response(400)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        import numpy as np
        import soundfile as sf
        samples, sample_rate = koko().create(text, voice=voice, speed=speed, lang="en-us")
        buf = io.BytesIO()
        sf.write(buf, np.array(samples), sample_rate, format="WAV")
        out = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(out)))
        self._cors()
        self.end_headers()
        self.wfile.write(out)

if __name__ == "__main__":
    print("warming up Kokoro-ONNX...", flush=True)
    koko()
    print("Amanda voice server on :8090 — female voices only, default sarah", flush=True)
    HTTPServer(("127.0.0.1", 8090), Handler).serve_forever()
