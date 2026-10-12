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
import asyncio, hashlib, hmac, io, json, os, struct, threading, time

import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from beryl_turns import Segmenter, TurnTaker
from beryl_face import _resample, unpack_stream

PORT = int(os.environ.get("BERYL_PORT", "8882"))
DEVICE = os.environ.get("BERYL_DEVICE", "cuda:0")
TOKEN = os.environ.get("BERYL_TOKEN", "")
FACE_URL = os.environ.get("BERYL_FACE_URL", "http://127.0.0.1:8883").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
# BERYL_PROFILE=lean uses the 0.6B ears (about 3 GB less) so ears, voice, brain and face fit together on a 24 GB card (L4).
LEAN = os.environ.get("BERYL_PROFILE", "full") == "lean"
ASR_MODEL = os.environ.get("BERYL_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B" if LEAN else "Qwen/Qwen3-ASR-1.7B")
TTS_MODEL = os.environ.get("BERYL_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")  # her voice stays full size in every profile
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


class Speech(BaseModel):
    input: str
    model: str = "qwen3tts"
    voice: str = "beryl"
    speed: float = 1.0
    face: bool = False          # also render her face for this sentence; the reply carries X-Utterance-Id


class RemoteFace:
    """The face process (beryl_face_server.py, FlashHead) on 127.0.0.1:8883."""

    def __init__(self, url=FACE_URL):
        self.url, self.up, self.checked, self.idle_cache = url, False, 0.0, None

    async def ready(self):
        if time.time() - self.checked > 10:
            self.checked = time.time()
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    self.up = (await c.get(self.url + "/health")).json().get("ready", False)
            except (httpx.HTTPError, ValueError):
                self.up = False
        return self.up

    async def idle(self):
        if self.idle_cache is None and await self.ready():
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(self.url + "/idle")
                if r.status_code == 200:
                    self.idle_cache = unpack_stream(r.content)[0]
        return self.idle_cache or []

    async def render(self, pcm16_bytes):
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", self.url + "/render", content=pcm16_bytes) as r:
                buf = b""
                async for chunk in r.aiter_bytes():
                    records, buf = unpack_stream(buf + chunk)
                    for rec in records:
                        yield rec


def frame_msg(kind, uid, t, jpeg):
    """Browser frame message: kind (1 idle, 2 utterance frame, 3 utterance end) | utterance id | length | seconds | JPEG."""
    return struct.pack(">BIIf", kind, uid, len(jpeg), float(t)) + jpeg


def trim_silence(wav, sr, lead_ms=70, trail_ms=140):
    """TTS output starts with dead air (measured 0.64 s on the real model). Cut it to a natural breath."""
    y = np.asarray(wav, dtype=np.float32)
    hop = max(1, sr // 100)
    n = len(y) // hop
    if n < 3:
        return y
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    live = np.where(rms > 0.04 * rms.max())[0]
    if len(live) == 0:
        return y
    a = max(0, live[0] * hop - int(sr * lead_ms / 1000))
    b = min(len(y), (live[-1] + 1) * hop + int(sr * trail_ms / 1000))
    return y[a:b]


BACKCHANNEL_TEXT = {"mm-hm": "Mm-hm.", "yeah": "Yeah.", "right": "Right.", "okay": "Okay.", "mm": "Mm.", "oh": "Oh."}


def _dtype():
    import torch
    if not DEVICE.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 (Turing) has no bf16


class Models:
    """Loads each model once and keeps it on the GPU. warm() does all of it at boot so the first sentence is not slow."""

    def __init__(self):
        self.lock = threading.Lock()
        self.asr = self.tts = self.prompt = None
        self.ready = {"ears": False, "voice": False, "clips": False}
        self.clips = {}
        self.error = ""

    def _load_asr(self):
        if self.asr is None:
            import torch
            from qwen_asr import Qwen3ASRModel
            self.asr = Qwen3ASRModel.from_pretrained(ASR_MODEL, dtype=_dtype(), device_map=DEVICE,
                                                     max_inference_batch_size=4, max_new_tokens=256)

    def _load_tts(self):
        if self.tts is None:
            from qwen_tts import Qwen3TTSModel
            self.tts = Qwen3TTSModel.from_pretrained(TTS_MODEL, device_map=DEVICE, dtype=_dtype())
        if self.prompt is None:
            self.prompt = self._voice()

    def transcribe(self, audio, lang):
        with self.lock:
            self._load_asr()
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
        with self.lock:
            self._load_tts()
            wavs, sr = self.tts.generate_voice_clone(text=text, language="English", voice_clone_prompt=self.prompt)
        return trim_silence(wavs[0], sr), sr

    def clip(self, kind):
        return self.clips.get(kind)

    def warm(self):
        """Load everything and run one throwaway sentence and one throwaway listen, so kernels are compiled and cached."""
        try:
            self.transcribe(np.zeros(SR, np.float32), "English")
            self.ready["ears"] = True
            self.speak("Okay.")
            self.ready["voice"] = True
            for kind, text in BACKCHANNEL_TEXT.items():
                self.clips[kind] = self.speak(text)
            self.ready["clips"] = True
        except Exception as e:  # surfaced through /health instead of dying silently
            self.error = f"{type(e).__name__}: {e}"


def make_app(models=None, token=None, face=None):
    models = models or Models()
    face = face if face is not None else (RemoteFace() if os.environ.get("BERYL_FACE", "1") != "0" else None)
    viewers, counter = set(), {"n": 0}
    token = TOKEN if token is None else token
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], expose_headers=["X-Utterance-Id"])

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
        ready = getattr(models, "ready", {"ears": True, "voice": True, "clips": True})
        return {"ok": True, "ready": all(ready.values()), "ears_ready": ready["ears"], "voice_ready": ready["voice"], "clips_ready": ready["clips"],
                "error": getattr(models, "error", ""), "asr": ASR_MODEL, "tts": TTS_MODEL, "profile": "lean" if LEAN else "full"}

    async def speech(req: Speech, tok: str = ""):
        guard(tok)
        text = req.input.strip()[:2000]
        if not text:
            raise HTTPException(400, "empty input")
        audio, sr = await asyncio.to_thread(models.speak, text)
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
        headers = {}
        if req.face and face is not None and viewers and await face.ready():
            counter["n"] += 1
            uid = counter["n"]
            pcm16 = (np.clip(_resample(audio, sr, 16000), -1, 1) * 32767).astype(np.int16).tobytes()
            asyncio.create_task(relay(uid, pcm16))
            headers["X-Utterance-Id"] = str(uid)
        return Response(buf.getvalue(), media_type="audio/wav", headers=headers)

    async def broadcast(msg):
        for ws in list(viewers):
            try:
                await ws.send_bytes(msg)
            except Exception:
                viewers.discard(ws)

    async def relay(uid, pcm16):
        try:
            async for t, jpeg in face.render(pcm16):
                await broadcast(frame_msg(2, uid, t, jpeg))
        except Exception as e:
            print(f"[beryl] face render failed for utterance {uid}: {e}")
        finally:
            await broadcast(frame_msg(3, uid, 0.0, b""))

    async def avatar(ws: WebSocket, tok: str = ""):
        if ws.headers.get("origin") not in ORIGINS or not allowed(tok) or face is None:
            await ws.close(code=1008)
            return
        await ws.accept()
        idle = await face.idle()
        if not idle:
            await ws.send_text(json.dumps({"type": "face", "ready": False}))
            await ws.close()
            return
        for t, jpeg in idle:
            await ws.send_bytes(frame_msg(1, 0, t, jpeg))
        await ws.send_text(json.dumps({"type": "face", "ready": True, "idle_frames": len(idle)}))
        viewers.add(ws)
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            viewers.discard(ws)

    async def backchannel(kind: str, tok: str = ""):
        guard(tok)
        item = models.clip(kind)
        if item is None:
            raise HTTPException(404, "clip not ready")
        audio, sr = item
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})

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
        seg, taker, lang = Segmenter(), TurnTaker(), "English"
        fed = 0

        async def send(obj):
            await ws.send_text(json.dumps(obj))

        async def hear(audio):
            return await asyncio.to_thread(models.transcribe, audio, lang)

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
                for ev in seg.feed(pcm):
                    kind = ev[0]
                    fed_s = (fed + len(pcm)) / SR
                    if kind == "start":
                        await send({"type": "speech", "state": "start"})
                    elif kind == "partial":
                        text = await hear(ev[1])
                        if text:
                            await send({"type": "partial", "text": text})
                    elif kind == "pause":
                        text = await hear(ev[1])
                        decision = taker.decide(text, seg.utterance_seconds, fed_s)
                        seg.set_hold(decision["hold_ms"])
                        if text:
                            await send({"type": "partial", "text": text, "hold_ms": decision["hold_ms"]})
                        if decision["backchannel"]:
                            await send({"type": "backchannel", "kind": decision["backchannel"]})
                    elif kind == "final":
                        text = await hear(ev[1])
                        await send({"type": "speech", "state": "end"})
                        if text:
                            await send({"type": "final", "text": text})
                fed += len(pcm)
        except WebSocketDisconnect:
            pass

    if hasattr(models, "warm") and os.environ.get("BERYL_WARM", "1") != "0":
        @app.on_event("startup")
        async def warm_on_boot():
            threading.Thread(target=models.warm, daemon=True).start()

    for prefix in ("", "/k/{tok}"):
        app.add_api_route(prefix + "/health", health, methods=["GET"])
        app.add_api_route(prefix + "/v1/audio/speech", speech, methods=["POST"])
        app.add_api_route(prefix + "/v1/backchannel/{kind}", backchannel, methods=["GET"])
        app.add_api_route(prefix + "/v1/models", llm_models, methods=["GET"])
        app.add_api_route(prefix + "/v1/chat/completions", llm_chat, methods=["POST"])
        app.add_api_websocket_route(prefix + "/v1/listen", listen)
        app.add_api_websocket_route(prefix + "/v1/avatar", avatar)
    return app


if __name__ == "__main__":
    import uvicorn
    print(f"[beryl] http://127.0.0.1:{PORT} profile={'lean' if LEAN else 'full'}  loading models now; /health says ready when done")
    uvicorn.run(make_app(), host="127.0.0.1", port=PORT, log_level="warning")
