"""Crane Voice: local OpenAI-compatible speech server for Crane Builder.

POST /v1/audio/speech  {"model": "kokoro"|"vibevoice", "voice": "af_heart", "input": "...", "speed": 1.0,
                        "voice2": "af_nicole", "mix": 0.25}   -> audio/wav (24 kHz)
GET  /health

kokoro     Kokoro-82M. Real-time on CPU. voice2/mix blends a second voice into the first.
vibevoice  microsoft/VibeVoice-Realtime-0.5B, built for live conversational speech (~300 ms to first
           audio on a T4/L4). Voices are VibeVoice's preset prompts, e.g. en-Emma_woman, en-Grace_woman.
           Install: git clone https://github.com/microsoft/VibeVoice next to this file (or set
           VIBEVOICE_DIR), then pip install -e "VibeVoice[streamingtts]". Best on a GPU.

Run:  pip install kokoro soundfile fastapi uvicorn
      python voice_server.py            # listens on 127.0.0.1:8880
"""
import io, os, threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

SR = 24000
PORT = int(os.environ.get("CRANE_VOICE_PORT", "8880"))
ORIGINS = [
    "https://aibruh-crane-builder-v1.static.hf.space",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.middleware("http")
async def private_network(request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


class Speech(BaseModel):
    input: str
    model: str = "kokoro"
    voice: str = "af_heart"
    voice2: str = ""
    mix: float = 0.0
    speed: float = 1.0


_lock = threading.Lock()
_kokoro = {}
_vv = {}
VV_MODEL = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
VV_DIR = os.environ.get("VIBEVOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "VibeVoice"))


def kokoro_say(req):
    from kokoro import KPipeline
    lang = req.voice[:1] if req.voice[:1] in "ab" else "a"
    if lang not in _kokoro:
        _kokoro[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    pipe = _kokoro[lang]
    voice = req.voice
    mix = max(0.0, min(1.0, req.mix))
    if req.voice2 and mix > 0:
        voice = pipe.load_voice(req.voice) * (1 - mix) + pipe.load_voice(req.voice2) * mix
    parts = [a.cpu().numpy() if hasattr(a, "cpu") else a for _, _, a in pipe(req.input, voice=voice, speed=req.speed)]
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def vibevoice_say(req):
    import copy, glob
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    if "model" not in _vv:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _vv["dev"] = dev
        _vv["proc"] = VibeVoiceStreamingProcessor.from_pretrained(VV_MODEL)
        kw = dict(torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev)
        try:
            m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(VV_MODEL, attn_implementation="flash_attention_2" if dev == "cuda" else "sdpa", **kw)
        except Exception:
            m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(VV_MODEL, attn_implementation="sdpa", **kw)
        m.eval()
        m.set_ddpm_inference_steps(num_steps=5)
        _vv["model"] = m
        _vv["voices"] = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in glob.glob(os.path.join(VV_DIR, "demo", "voices", "streaming_model", "**", "*.pt"), recursive=True)}
        _vv["prompts"] = {}
    name = req.voice.lower()
    if name not in _vv["voices"]:
        raise HTTPException(400, f"unknown VibeVoice voice {req.voice!r}; set VIBEVOICE_DIR to your VibeVoice clone")
    if name not in _vv["prompts"]:
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            _vv["prompts"][name] = torch.load(_vv["voices"][name], map_location=_vv["dev"], weights_only=True)
    prompt = _vv["prompts"][name]
    text = req.input.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    inputs = _vv["proc"].process_input_with_cached_prompt(text=text, cached_prompt=prompt, padding=True, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(_vv["dev"]) if torch.is_tensor(v) else v for k, v in inputs.items()}
    out = _vv["model"].generate(**inputs, max_new_tokens=None, cfg_scale=1.5, tokenizer=_vv["proc"].tokenizer,
                                generation_config={"do_sample": False}, verbose=False, all_prefilled_outputs=copy.deepcopy(prompt))
    if not out.speech_outputs or out.speech_outputs[0] is None:
        raise HTTPException(500, "VibeVoice produced no audio")
    return out.speech_outputs[0].float().cpu().numpy().reshape(-1)


@app.get("/health")
def health():
    return {"ok": True, "engines": ["kokoro", "vibevoice"]}


@app.post("/v1/audio/speech")
def speech(req: Speech):
    text = req.input.strip()[:2000]
    if not text:
        raise HTTPException(400, "empty input")
    req.input = text
    engine = req.model.lower()
    with _lock:
        if engine.startswith("kokoro"):
            audio = kokoro_say(req)
        elif engine.startswith("vibevoice"):
            audio = vibevoice_say(req)
        else:
            return JSONResponse({"error": f"engine {req.model!r} is not served here (kokoro, vibevoice)"}, status_code=501)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    print(f"[crane-voice] http://127.0.0.1:{PORT}/v1/audio/speech")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
