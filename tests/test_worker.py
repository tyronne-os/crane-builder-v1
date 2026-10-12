"""Worker behavior tests with fake models: turn-taking over the real /v1/listen socket, backchannel clips, dead-air trim.
Run:  python tests/test_worker.py   (needs fastapi, httpx, numpy, soundfile; no GPU)"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ['BERYL_WARM'] = '0'
import beryl_gpu_server as B
import beryl_turns as T
from fastapi.testclient import TestClient

rng = np.random.default_rng(2)
fails = 0


def check(name, cond, extra=''):
    global fails
    fails += (not cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if (extra and not cond) else ''))


def speech(sec):
    n = int(16000 * sec)
    return (0.25 * rng.standard_normal(n) * np.abs(np.sin(np.linspace(0, 30 * sec, n)))).astype(np.float32)


def quiet(sec):
    return (0.0008 * rng.standard_normal(int(16000 * sec))).astype(np.float32)


class Scripted:
    """Fake ASR: returns whatever the test registered for utterances of a given length."""
    def __init__(self, script): self.script, self.ready = script, {"ears": True, "voice": True, "clips": True}
    error = ''
    def transcribe(self, audio, lang):
        s = len(audio) / 16000
        for limit, text in self.script:
            if s <= limit:
                return text
        return self.script[-1][1]
    def speak(self, text): return np.zeros(2400, np.float32), 24000
    def clip(self, kind): return (0.3 * np.sin(np.linspace(0, 400, 12000)).astype(np.float32), 24000) if kind in T.BACKCHANNELS else None


def session(models, token='tok'):
    B.TurnTaker = lambda: T.TurnTaker(seed=1, backchannel_chance=1.0)
    return TestClient(B.make_app(models, token=token))


import asyncio


async def drive(models, parts):
    """Runs the worker's real listen() coroutine against an in-memory socket and records (stream_seconds, message)."""
    app = B.make_app(models, token='tok')
    route = next(r for r in app.routes if getattr(r, 'path', '') == '/k/{tok}/v1/listen')
    log, inbox = [], asyncio.Queue()
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)
    now = {'t': 0.0}

    class WS:
        headers = {'origin': 'http://localhost:8000'}
        async def accept(self): pass
        async def close(self, code=1000): pass
        async def send_text(self, s): log.append((round(now['t'], 2), json.loads(s)))
        async def receive(self):
            item = await inbox.get()
            return item

    ws = WS()
    task = asyncio.create_task(route.endpoint(ws, tok='tok'))
    await inbox.put({'text': json.dumps({'type': 'config', 'lang': 'English'})})
    for i in range(0, len(pcm), 1600):
        now['t'] = (i + 1600) / 16000
        await inbox.put({'bytes': pcm[i:i + 1600].tobytes()})
        await asyncio.sleep(0)
        while not inbox.empty():
            await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    await inbox.put({'type': 'websocket.disconnect'})
    await task
    return log


def times(log, typ, **kw):
    return [t for t, m in log if m.get('type') == typ and all(m.get(k) == v for k, v in kw.items())]


B.TurnTaker = lambda: T.TurnTaker(seed=1, backchannel_chance=1.0)

# 1. A long turn with a mid-clause pause gets a nod, and the turn is NOT taken at that pause.
script = [(2.0, 'So I started with'), (6.8, 'So I started with the landing page, and then I moved the headline up,'),
          (99, 'So I started with the landing page, and then I moved the headline up, and now it looks great.')]
log = asyncio.run(drive(Scripted(script), [quiet(0.5), speech(6.5), quiet(0.55), speech(3.0), quiet(2.0)]))
bc = times(log, 'backchannel')
fin = [(t, m['text']) for t, m in log if m['type'] == 'final']
check('long turn at a clause boundary: backchannel arrives during the pause', len(bc) == 1 and 7.0 <= bc[0] <= 7.4, bc)
check('the nod does not end the turn: one final, after the second burst', len(fin) == 1 and fin[0][0] > 10.0, fin)
check('the final carries the whole turn', fin and fin[0][1].endswith('now it looks great.'), fin)
check('speech start and end are announced', len(times(log, 'speech', state='start')) == 1 and len(times(log, 'speech', state='end')) == 1)

# 2. Pause after "and": hold the floor open past the default 700 ms.
log = asyncio.run(drive(Scripted([(99, 'I want the button to be gold and')]), [quiet(0.5), speech(2.0), quiet(1.2)]))
check('mid-thought pause (ends on "and"): no final within 1.2 s of silence', not times(log, 'final'), log[-2:])
log = asyncio.run(drive(Scripted([(99, 'I want the button to be gold and')]), [quiet(0.5), speech(2.0), quiet(1.8)]))
check('...but the turn does end once the longer hold passes', len(times(log, 'final')) == 1)

# 3. Clear question: respond quickly (hold about 300 ms, not 700).
log = asyncio.run(drive(Scripted([(99, 'Can you make it bigger?')]), [quiet(0.5), speech(2.0), quiet(1.0)]))
f = times(log, 'final')
check('question: turn ends about 300-400 ms after speech stops', len(f) == 1 and 0.28 <= f[0] - 2.5 <= 0.55, f)

# 4. Backchannel clip route.
c = session(Scripted([(99, 'x')]))
r = c.get('/k/tok/v1/backchannel/mm-hm')
check('backchannel clip is served as wav with the token', r.status_code == 200 and r.headers['content-type'] == 'audio/wav' and r.content[:4] == b'RIFF')
check('unknown clip is 404', c.get('/k/tok/v1/backchannel/nope').status_code == 404)
check('clip route hidden without the token', c.get('/v1/backchannel/mm-hm').status_code == 404)

# 5. Dead-air trim.
sr = 24000
tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
y = np.concatenate([np.zeros(int(0.64 * sr), np.float32), tone, np.zeros(int(0.5 * sr), np.float32)])
z = B.trim_silence(y, sr)
check('dead air trimmed: 0.64 s lead and 0.5 s tail become a breath', 1.15 <= len(z) / sr <= 1.30, round(len(z) / sr, 2))
check('trim keeps all of the speech', float(np.max(np.abs(z))) > 0.29 and abs(np.sum(np.abs(z)) - np.sum(np.abs(tone))) / np.sum(np.abs(tone)) < 0.02)
check('silent input does not crash', len(B.trim_silence(np.zeros(1000, np.float32), sr)) == 1000)

print('ALL PASS' if not fails else f'{fails} FAILED'); sys.exit(1 if fails else 0)
