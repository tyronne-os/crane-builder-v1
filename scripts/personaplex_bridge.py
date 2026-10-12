"""NVIDIA PersonaPlex (full-duplex speech model) behind the Beryl worker.

PersonaPlex listens and talks at the same time (interruptions, "mm-hm", overlap) the way Tavus' Griffin does, but it is a
7B conversation model, not a builder: it cannot write code or drive the computer. So the bridge runs two things on the same
microphone stream:

  * PersonaPlex      speaks and listens in real time, in Beryl's voice (her designed voice is its voice prompt).
  * Qwen3-ASR        transcribes what the user said; the Router decides whether it is chat (PersonaPlex keeps going) or a
                     request to build / research / use the computer. For those it sends {"type":"handoff"}: PersonaPlex's
                     audio is muted, the builder's Qwen brain answers through Qwen3-TTS in the same voice, and the browser
                     sends {"type":"resume"} when she has finished.

DuplexEngine mirrors NVIDIA's offline script (github.com/NVIDIA/personaplex, moshi/moshi/offline.py) call for call. It needs
the `moshi` package from that repo, the model license accepted on Hugging Face, and an A100/H100-class GPU for real time
(NVIDIA's tested hardware); it has not been run on any GPU yet. DuplexSession is the protocol around it and is unit-tested
with a fake engine (tests/test_duplex.py).
"""
import asyncio
import json
import re

import numpy as np

from beryl_turns import Segmenter

MODEL_SR = 24000
FRAME24 = 1920          # 80 ms at 24 kHz: one Mimi frame (12.5 Hz)

# PersonaPlex was trained on casual "You enjoy having a good conversation." prompts plus a scenario; keep to that shape.
PERSONA_PROMPT = (
    "You enjoy having a good conversation. Your name is Beryl. You are warm, quick-witted and genuinely curious, a close "
    "friend who also happens to be great at building websites. You tease a little, laugh easily, and react with your whole "
    "voice: 'oh', 'hm', 'wait, really?'. Keep your turns short, one to three sentences, and ask a question only when you "
    "actually want the answer. If they ask you to build, change, fix, search for or open something, say you're on it in a "
    "few words and stop talking; a teammate handles the building."
)

_ACTION_VERBS = r"(build|make|create|change|add|remove|delete|fix|update|edit|move|resize|rename|deploy|push|publish|ship|open|launch|search|look up|google|research|find|show me|write|generate|install|run|test|set up|switch|turn (it|that|this) into|paste)"
_OBJECTS = r"(page|site|website|button|header|footer|nav|navbar|menu|logo|image|photo|color|colour|font|layout|section|landing|app|code|script|function|bug|file|repo|github|hugging ?face|space|paper|model|chrome|browser|tab|link|url|css|html|react|project|dark mode|theme)"
_PHRASES = (r"\b(can|could|would|will) you (please )?" + _ACTION_VERBS, r"\blet'?s (build|make|create|fix|add|change)\b",
            r"\b(go to|take me to|pull up|bring up|navigate to) ", r"^(please )?" + _ACTION_VERBS + r"\b.*\b" + _OBJECTS + r"\b")


class Router:
    """Chat stays with PersonaPlex; anything that needs hands goes to the builder. Conservative on purpose: a missed
    request makes PersonaPlex say "I can't do that", a false alarm mutes friendly chat for a moment."""

    def needs_builder(self, text):
        t = text.strip().lower()
        if not t:
            return False
        if re.search(r"https?://|\bwww\.|\.(com|io|dev|ai|co)\b", t):
            return True
        for p in _PHRASES:
            if re.search(p, t):
                return bool(re.search(_OBJECTS, t) or re.search(r"\b(for|up|me) ", t) or re.search(r"\b(go to|take me to|pull up|navigate to)\b", t))
        return False


def resample(x, src, dst):
    """Linear resample, good enough for speech going into a codec that is itself lossy."""
    if src == dst or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class DuplexEngine:
    """PersonaPlex-7B in process. Mirrors moshi/offline.py: Mimi x2, Moshi LM, LMGen, voice + text prompt, then 80 ms steps."""

    def __init__(self, hf_repo="nvidia/personaplex-7b-v1", device="cuda:0", cpu_offload=False, seed=None):
        self.hf_repo, self.device, self.cpu_offload, self.seed = hf_repo, device, cpu_offload, seed
        self.loaded = False

    def load(self):
        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders, LMGen
        if self.seed is not None:
            torch.manual_seed(self.seed)
        mimi_w = hf_hub_download(self.hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_w, self.device)
        self.other_mimi = loaders.get_mimi(mimi_w, self.device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(hf_hub_download(self.hf_repo, loaders.TEXT_TOKENIZER_NAME))
        lm = loaders.get_moshi_lm(hf_hub_download(self.hf_repo, loaders.MOSHI_NAME), device=self.device, cpu_offload=self.cpu_offload)
        lm.eval()
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate), sample_rate=self.mimi.sample_rate,
                            device=self.device, frame_rate=self.mimi.frame_rate, save_voice_prompt_embeddings=False,
                            use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
        for m in (self.mimi, self.other_mimi, self.lm_gen):
            m.streaming_forever(1)
        with torch.no_grad():
            for _ in range(4):
                chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
                codes = self.mimi.encode(chunk)
                self.other_mimi.encode(chunk)
                for c in range(codes.shape[-1]):
                    tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                    if tokens is not None:
                        self.mimi.decode(tokens[:, 1:9])
                        self.other_mimi.decode(tokens[:, 1:9])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.loaded = True

    def start(self, voice_prompt, text_prompt):
        """voice_prompt: a .wav (Beryl's designed voice) or a packaged embedding such as NATF2.pt. Starts a fresh conversation."""
        import torch
        if voice_prompt.endswith(".pt"):
            self.lm_gen.load_voice_prompt_embeddings(voice_prompt)
        else:
            self.lm_gen.load_voice_prompt(voice_prompt)
        text = text_prompt.strip()
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(text if text.startswith("<system>") else f"<system> {text} <system>") if text else None
        with torch.no_grad():
            self.mimi.reset_streaming(); self.other_mimi.reset_streaming(); self.lm_gen.reset_streaming()
            self.lm_gen.step_system_prompts(self.mimi)
            self.mimi.reset_streaming()

    def feed(self, frame24):
        """One 80 ms frame of user audio in (float32, 24 kHz, 1920 samples). Returns [(agent_pcm_float32 or None, text_piece or None)]."""
        import torch
        out = []
        with torch.no_grad():
            chunk = torch.from_numpy(frame24.astype(np.float32))[None, None].to(self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is None:
                    continue
                pcm = self.mimi.decode(tokens[:, 1:9])
                self.other_mimi.decode(tokens[:, 1:9])
                tid = tokens[0, 0, 0].item()
                piece = None if tid in (0, 3) else self.tokenizer.id_to_piece(tid).replace("▁", " ") if tid > 3 else None
                out.append((pcm.detach().cpu().numpy()[0, 0], piece))
        return out


class DuplexSession:
    """One browser connection. `engine` is a DuplexEngine (or a fake); `hear(audio16k) -> text` is Qwen3-ASR."""

    def __init__(self, engine, hear, send_json, send_bytes, router=None, default_voice="NATF2.pt", voice_resolver=None):
        self.engine, self.hear, self.send_json, self.send_bytes = engine, hear, send_json, send_bytes
        self.router = router or Router()
        self.default_voice, self.voice_resolver = default_voice, voice_resolver
        self.seg, self.buf24, self.muted, self.started = Segmenter(), np.zeros(0, np.float32), False, False

    async def configure(self, cfg):
        voice = cfg.get("voice") or self.default_voice
        if self.voice_resolver:
            voice = self.voice_resolver(voice)
        persona = cfg.get("persona") or PERSONA_PROMPT
        await asyncio.to_thread(self.engine.start, voice, persona)
        self.started = True
        await self.send_json({"type": "ready", "voice": voice})

    async def message(self, cfg):
        kind = cfg.get("type")
        if kind == "config":
            await self.configure(cfg)
        elif kind == "resume":
            self.muted = False
            await self.send_json({"type": "resumed"})
        elif kind == "ping":
            await self.send_json({"type": "pong"})

    async def audio(self, pcm16_bytes):
        if not self.started:
            await self.configure({})
        x16 = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        for ev in self.seg.feed(x16):
            if ev[0] == "start":
                await self.send_json({"type": "speech", "state": "start"})
            elif ev[0] == "final":
                text = await asyncio.to_thread(self.hear, ev[1])
                await self.send_json({"type": "speech", "state": "end"})
                if text:
                    await self.send_json({"type": "final", "text": text})
                    if self.router.needs_builder(text):
                        self.muted = True
                        await self.send_json({"type": "handoff", "text": text})
        self.buf24 = np.concatenate([self.buf24, resample(x16, 16000, MODEL_SR)])
        while len(self.buf24) >= FRAME24:
            frame, self.buf24 = self.buf24[:FRAME24], self.buf24[FRAME24:]
            for pcm, piece in await asyncio.to_thread(self.engine.feed, frame):
                if piece and not self.muted:
                    await self.send_json({"type": "agent_text", "text": piece})
                if pcm is not None and not self.muted:
                    await self.send_bytes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
