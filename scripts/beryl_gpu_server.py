"""Beryl GPU server: Qwen3-ASR ears + Qwen3-TTS voice for Crane Builder, on one GPU.

GET  /health
POST /v1/audio/speech   {"input": "...", "model": "qwen3tts", "voice": "beryl"}  -> audio/wav
POST /v1/chat/completions, GET /v1/models   passed through to Ollama (the brain) at OLLAMA_URL
Set BERYL_TOKEN (any long random string) before exposing this through a tunnel: every route is then served only under
/k/<token>/v1/..., so one URL carries its own secret (put https://host/k/<token>/v1 in the builder; wss for listening).
WS   /v1/listen         browser sends {"type":"config","lang":"English"} then 16 kHz mono PCM16 frames;
                        server replies {"type":"partial","text"} while you talk and {"type":"final","text"} when you stop.

Beryl's voice is designed once with Qwen3-TTS-VoiceDesign from BERYL_VOICE_PROMPT, saved to voices/beryl-<key>.wav,
then every reply clones that clip with Qwen3-TTS-Base so her voice never drifts.

Run on the GPU machine:  python beryl_gpu_server.py   (listens on 127.0.0.1:8882; reach it through the SSH tunnel)
"""
import asyncio, hashlib, hmac, io, json, os, threading

import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

PORT = int(os.environ.get("BERYL_PORT", "8882"))
DEVICE = os.environ.get("BERYL_DEVICE", "cuda:0")
TOKEN = os.environ.get("BERYL_TOKEN", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
ASR_MODEL = os.environ.get("BERYL_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
TTS_MODEL = os.environ.get("BERYL_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DESIGN_MODEL = os.environ.get("BERYL_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VOICE_DIR = os.environ.get("BERYL_VOICE_DIR", os.path.join(HERE, "voices"))
# Voice design brief. Qwen3-TTS-VoiceDesign reads attributes best: who, timbre, pitch, mic distance, pace, emotion, what to avoid.
VOICE_PROMPT = os.environ.get("BERYL_VOICE_PROMPT", (
    "Female, early thirties, American English with a soft neutral accent. Mezzo-soprano sitting low in her range, "
    "warm and slightly husky, with a gentle breath at the ends of phrases. Close-mic and intimate, like she is sitting "
    "across a small table from you. Relaxed, unhurried pace that quickens when she gets excited, lively natural pitch "
    "movement, audibly smiling. Curious, playful and emotionally present. Not an announcer, not a narrator, no vocal fry, "
    "never flat or robotic."))
# The clone copies the reference clip's prosody, so the line carries hesitation, a laugh, warmth and a real question.
VOICE_LINE = os.environ.get("BERYL_VOICE_LINE", (
    "Hm, okay... so here's the thing. I actually love this idea, like, a lot. Ha, I know, I get excited. "
    "But wait, tell me, what made you think of it?"))
VOICE_KEY = hashlib.sha1((VOICE_PROMPT + "|" + VOICE_LINE).encode()).hexdigest()[:8]
ORIGINS = [
    "https://aibruh-crane-builder-v1.static.hf.space",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
SR = 16000
FRAME = SR // 50  # 20 ms


class Speech(BaseModel):
    input: str
    model: str = "qwen3tts"
    voice: str = "beryl"
    speed: float = 1.0


class Segmenter:
    """Energy voice detection: turns a PCM stream into utterances, with partial snapshots while you talk."""

    def __init__(self, start_ms=60, end_ms=int(os.environ.get("BERYL_END_SILENCE_MS", "700")),
                 preroll_ms=300, partial_ms=600, max_s=20):
        self.start_n, self.end_n = start_ms // 20, end_ms // 20
        self.preroll, self.partial_n, self.max_n = preroll_ms // 20, partial_ms // 20, max_s * 50
        self.floor, self.pending = 0.003, np.zeros(0, np.float32)
        self.ring, self.utt, self.talking, self.loud, self.quiet, self.since = [], [], False, 0, 0, 0

    def feed(self, pcm):
        """Yields ("partial", audio) and ("final", audio) events."""
        self.pending = np.concatenate([self.pending, pcm])
        while len(self.pending) >= FRAME:
            f, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            rms = float(np.sqrt(np.mean(f * f)))
            voiced = rms > max(self.floor * 3.0, 0.008)
            if not self.talking:
                self.floor = 0.95 * self.floor + 0.05 * min(rms, 0.05) if not voiced else self.floor
                self.ring = (self.ring + [f])[-self.preroll:]
                self.loud = self.loud + 1 if voiced else 0
                if self.loud >= self.start_n:
                    self.talking, self.utt, self.quiet, self.since = True, list(self.ring), 0, 0
                continue
            self.utt.append(f)
            self.quiet = 0 if voiced else self.quiet + 1
            self.since += 1
            if self.quiet >= self.end_n or len(self.utt) >= self.max_n:
                audio = np.concatenate(self.utt)
                self.talking, self.utt, self.ring, self.loud = False, [], [], 0
                yield "final", audio
            elif self.since >= self.partial_n:
                self.since = 0
                yield "partial", np.concatenate(self.utt)


def _dtype():
    import torch
    if not DEVICE.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 (Turing) has no bf16


class Models:
    """Loads each model on first use and keeps it on the GPU."""

    def __init__(self):
        self.lock = threading.Lock()
        self.asr = self.tts = self.prompt = None

    def transcribe(self, audio, lang):
        import torch
        from qwen_asr import Qwen3ASRModel
        with self.lock:
            if self.asr is None:
                self.asr = Qwen3ASRModel.from_pretrained(ASR_MODEL, dtype=_dtype(), device_map=DEVICE,
                                                         max_inference_batch_size=4, max_new_tokens=256)
            r = self.asr.transcribe(audio=(audio, SR), language=lang or None)
        return r[0].text.strip()

    def _voice(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        wav, txt = os.path.join(VOICE_DIR, f"beryl-{VOICE_KEY}.wav"), os.path.join(VOICE_DIR, f"beryl-{VOICE_KEY}.txt")
        if not os.path.exists(wav):
            os.makedirs(VOICE_DIR, exist_ok=True)
            design = Qwen3TTSModel.from_pretrained(DESIGN_MODEL, device_map=DEVICE, dtype=_dtype())
            ref, sr = design.generate_voice_design(text=VOICE_LINE, language="English", instruct=VOICE_PROMPT)
            sf.write(wav, ref[0], sr)
            open(txt, "w").write(VOICE_LINE)
            del design
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self.tts.create_voice_clone_prompt(ref_audio=wav, ref_text=open(txt).read().strip())

    def speak(self, text):
        import torch
        from qwen_tts import Qwen3TTSModel
        with self.lock:
            if self.tts is None:
                self.tts = Qwen3TTSModel.from_pretrained(TTS_MODEL, device_map=DEVICE, dtype=_dtype())
            if self.prompt is None:
                self.prompt = self._voice()
            wavs, sr = self.tts.generate_voice_clone(text=text, language="English", voice_clone_prompt=self.prompt)
        return wavs[0], sr


def make_app(models=None, token=None):
    models = models or Models()
    token = TOKEN if token is None else token
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

    @app.middleware("http")
    async def private_network(request, call_next):
        resp = await call_next(request)
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
        return resp

    def allowed(tok):
        # With a token set, only /k/<token>/... works; wrong or missing token looks like a missing page.
        return hmac.compare_digest(tok.encode(), token.encode()) if token else tok == ""

    def guard(tok):
        if not allowed(tok):
            raise HTTPException(404, "Not Found")

    async def health(tok: str = ""):
        guard(tok)
        return {"ok": True, "asr": ASR_MODEL, "tts": TTS_MODEL,
                "voice_ready": os.path.exists(os.path.join(VOICE_DIR, f"beryl-{VOICE_KEY}.wav"))}

    async def speech(req: Speech, tok: str = ""):
        guard(tok)
        text = req.input.strip()[:2000]
        if not text:
            raise HTTPException(400, "empty input")
        audio, sr = await asyncio.to_thread(models.speak, text)
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    async def llm_models(tok: str = ""):
        guard(tok)
        async with httpx.AsyncClient(timeout=15) as c:
            try:
                r = await c.get(OLLAMA_URL + "/models")
            except httpx.HTTPError:
                raise HTTPException(502, "brain (Ollama) is not running")
        return JSONResponse(r.json(), status_code=r.status_code)

    async def llm_chat(request: Request, tok: str = ""):
        guard(tok)
        body = await request.body()
        try:
            wants_stream = bool(json.loads(body).get("stream"))
        except ValueError:
            raise HTTPException(400, "bad JSON")

        async def pipe():
            async with httpx.AsyncClient(timeout=None) as c:
                try:
                    async with c.stream("POST", OLLAMA_URL + "/chat/completions", content=body,
                                        headers={"Content-Type": "application/json"}) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
                except httpx.HTTPError as e:
                    yield json.dumps({"error": {"message": "brain (Ollama) is not running: " + str(e)}}).encode()
        return StreamingResponse(pipe(), media_type="text/event-stream" if wants_stream else "application/json")

    async def listen(ws: WebSocket, tok: str = ""):
        if ws.headers.get("origin") not in ORIGINS or not allowed(tok):
            await ws.close(code=1008)
            return
        await ws.accept()
        seg, lang, busy = Segmenter(), "English", False
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text"):
                    cfg = json.loads(msg["text"])
                    if cfg.get("type") == "config":
                        lang = None if cfg.get("lang") in (None, "", "auto") else cfg["lang"]
                    continue
                pcm = np.frombuffer(msg.get("bytes") or b"", dtype=np.int16).astype(np.float32) / 32768.0
                for kind, audio in seg.feed(pcm):
                    if kind == "partial" and busy:
                        continue
                    busy = True
                    try:
                        text = await asyncio.to_thread(models.transcribe, audio, lang)
                    finally:
                        busy = False
                    if text:
                        await ws.send_text(json.dumps({"type": kind, "text": text}))
        except WebSocketDisconnect:
            pass

    for prefix in ("", "/k/{tok}"):
        app.add_api_route(prefix + "/health", health, methods=["GET"])
        app.add_api_route(prefix + "/v1/audio/speech", speech, methods=["POST"])
        app.add_api_route(prefix + "/v1/models", llm_models, methods=["GET"])
        app.add_api_route(prefix + "/v1/chat/completions", llm_chat, methods=["POST"])
        app.add_api_websocket_route(prefix + "/v1/listen", listen)
    return app


if __name__ == "__main__":
    import uvicorn
    print(f"[beryl] http://127.0.0.1:{PORT}  (speech: /v1/audio/speech, listening: ws /v1/listen)")
    uvicorn.run(make_app(), host="127.0.0.1", port=PORT, log_level="warning")
