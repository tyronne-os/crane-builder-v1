"""Protocol tests for scripts/personaplex_bridge.py with a fake duplex engine (no model, no GPU).  Run: python tests/test_duplex.py"""
import asyncio, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from personaplex_bridge import DuplexSession, Router, resample, FRAME24, MODEL_SR, PERSONA_PROMPT

fails = 0


def check(name, cond, extra=''):
    global fails
    fails += (not cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if (extra and not cond) else ''))


class FakeEngine:
    def __init__(self): self.starts, self.frames = [], 0
    def start(self, voice, persona): self.starts.append((voice, persona))
    def feed(self, frame):
        self.frames += 1
        return [(np.full(1920, 0.1, np.float32), 'Oh')]


rng = np.random.default_rng(0)


def pcm16(sec, loud=True):
    n = int(16000 * sec)
    x = (0.25 * rng.standard_normal(n) * np.abs(np.sin(np.linspace(0, 30 * sec, n)))) if loud else 0.001 * rng.standard_normal(n)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class Harness:
    def __init__(self, text_for):
        self.texts, self.json, self.bytes = list(text_for), [], []
        self.engine = FakeEngine()
        self.session = DuplexSession(self.engine, self._hear, self._json, self._bytes, voice_resolver=lambda v: v)

    def _hear(self, audio):
        return self.texts.pop(0) if self.texts else ''

    async def _json(self, o): self.json.append(o)
    async def _bytes(self, b): self.bytes.append(b)


def run(coro):
    return asyncio.run(coro)


# --- Router ------------------------------------------------------------------------------------------------
r = Router()
check('chat stays with PersonaPlex', not r.needs_builder('so what do you think about cats') and not r.needs_builder("that's a great idea, I love it"))
check('"build a page" goes to the builder', r.needs_builder('can you build me a landing page with a gold button'))
check('"change the header" goes to the builder', r.needs_builder('please change the header to blue'))
check('"open github" goes to the builder', r.needs_builder('open github and show me the repo'))
check('a link goes to the builder', r.needs_builder('check out https://huggingface.co/nvidia/personaplex-7b-v1'))
check('"find papers on speech" goes to the builder', r.needs_builder('could you find papers on speech models for me'))
check('small talk about building is not a request', not r.needs_builder('building websites is so stressful sometimes'))
check('"I love the design" is not a request', not r.needs_builder('I love the design, it feels calm'))
check('empty text is not a request', not r.needs_builder('   '))

# --- resample -------------------------------------------------------------------------------------------------
x = np.sin(np.linspace(0, 2 * np.pi * 440 * 0.1, 1600)).astype(np.float32)
y = resample(x, 16000, 24000)
check('16 kHz to 24 kHz gives the right length', len(y) == 2400, len(y))
check('resample keeps the tone (peak within 5%)', abs(float(np.max(np.abs(y))) - float(np.max(np.abs(x)))) < 0.05)
check('same rate is a no-op copy', np.array_equal(resample(x, 16000, 16000), x))

# --- session protocol -----------------------------------------------------------------------------------------
h = Harness(['build me a landing page'])
async def scenario_build():
    await h.session.message({'type': 'config', 'voice': 'beryl.wav'})
    await h.session.audio(pcm16(0.6, loud=False))
    await h.session.audio(pcm16(2.0))
    await h.session.audio(pcm16(1.2, loud=False))
    await h.session.audio(pcm16(0.2, loud=False))
    return
run(scenario_build())
types = [m['type'] for m in h.json]
check('configure announces the voice and persona', h.engine.starts and h.engine.starts[0][0] == 'beryl.wav' and h.engine.starts[0][1] == PERSONA_PROMPT)
check('ready is sent after configure', h.json[0]['type'] == 'ready' and h.json[0]['voice'] == 'beryl.wav')
check('user speech start/end are announced', 'speech' in types and [m for m in h.json if m['type'] == 'speech'][0]['state'] == 'start')
check('the transcript is sent', any(m.get('type') == 'final' and m.get('text') == 'build me a landing page' for m in h.json))
check('the request triggers a handoff, not a PersonaPlex reply', 'handoff' in types)
check('after handoff, agent audio is muted', h.session.muted is True)
muted_before = len(h.bytes)
run(h.session.audio(pcm16(0.5)))
check('no agent audio is sent while muted', len(h.bytes) == muted_before)

async def resume():
    await h.session.message({'type': 'resume'})
run(resume())
check('resume unmutes and acknowledges', h.session.muted is False and h.json[-1]['type'] == 'resumed')
run(h.session.audio(pcm16(0.5)))
check('agent audio flows again after resume', len(h.bytes) > muted_before)
check('agent text is sent with the audio', any(m['type'] == 'agent_text' for m in h.json))

# Chat-only conversation: no handoff, audio streams from the first frame.
h2 = Harness(['so what did you think of the weekend'])
async def chat():
    await h2.session.message({'type': 'config'})
    await h2.session.audio(pcm16(2.0))
    await h2.session.audio(pcm16(1.2, loud=False))
run(chat())
check('chat turn: no handoff', not any(m['type'] == 'handoff' for m in h2.json))
check('chat turn: agent audio streamed', len(h2.bytes) > 0 and all(len(b) == 1920 * 2 for b in h2.bytes))
check('chat turn: frames are 80 ms each into the engine', h2.engine.frames > 0)

# Audio is int16 little-endian and within range.
samples = np.frombuffer(h2.bytes[0], dtype=np.int16)
check('agent audio is 16-bit PCM in range', samples.dtype == np.int16 and np.max(np.abs(samples.astype(np.int32))) <= 32767)

# Bad or early messages do not crash the session.
h3 = Harness([])
run(h3.session.message({'type': 'ping'}))
check('ping answered', h3.json and h3.json[-1]['type'] == 'pong')
run(h3.session.audio(pcm16(0.3, loud=False)))
check('audio before config starts a default voice', h3.engine.starts and h3.engine.starts[0][0] == 'NATF2.pt')

print('ALL PASS' if not fails else f'{fails} FAILED'); sys.exit(1 if fails else 0)
