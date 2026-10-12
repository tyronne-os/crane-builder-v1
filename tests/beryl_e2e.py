"""End-to-end check of the Beryl worker + the builder, with fake models (no GPU needed).

Starts beryl_gpu_server.make_app() with a fake ASR/TTS and a fake Ollama, behind a secret-token URL, then drives
frontend/app.html in Chromium with a fake microphone. Run from the repo root:  python tests/beryl_e2e.py
Needs: pip install fastapi uvicorn[standard] soundfile numpy httpx playwright (+ playwright's chromium).
"""
import http.server, json, os, subprocess, sys, threading, time, urllib.request, wave
import numpy as np, uvicorn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import beryl_gpu_server as B
import beryl_turns as T
from playwright.sync_api import sync_playwright

TOKEN = 'sekret-token-123'
calls = {'transcribe': [], 'speak': [], 'brain': [], 'clips': []}


class FakeModels:
    def transcribe(self, audio, lang):
        s = len(audio) / 16000
        calls['transcribe'].append((round(s, 2), lang))
        return 'build me' if s < 1.2 else 'build me a landing page,' if s < 3.0 else 'build me a landing page, in gold.'

    def clip(self, kind):
        calls['clips'].append(kind)
        return (0.3 * np.sin(np.linspace(0, 300, 9600)).astype(np.float32), 24000) if kind in T.BACKCHANNELS else None

    def speak(self, text):
        calls['speak'].append(text)
        t = np.linspace(0, 1.2, int(24000 * 1.2), False)
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000


class FakeOllama(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        body = json.dumps({'data': [{'id': 'fake-qwen'}]}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        calls['brain'].append(req)
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        for piece in ['Oh, love it. ', 'Let me start on that.']:
            self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': {'content': piece}}]}) + '\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')


os.environ['OLLAMA_URL'] = 'http://127.0.0.1:11434/v1'
B.OLLAMA_URL = os.environ['OLLAMA_URL']
threading.Thread(target=http.server.ThreadingHTTPServer(('127.0.0.1', 11434), FakeOllama).serve_forever, daemon=True).start()
B.TurnTaker = lambda: T.TurnTaker(seed=2, backchannel_after_s=1.0, backchannel_chance=1.0)
threading.Thread(target=lambda: uvicorn.run(B.make_app(FakeModels(), token=TOKEN), host='127.0.0.1', port=8882, log_level='warning'), daemon=True).start()
web = subprocess.Popen([sys.executable, '-m', 'http.server', '8000', '-d', os.path.join(ROOT, 'frontend')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

sr = 48000
rng = np.random.default_rng(0)
def burst(sec): return 0.3 * rng.standard_normal(int(sr * sec)) * np.abs(np.sin(np.linspace(0, 40, int(sr * sec))))
mic = np.concatenate([0.001 * rng.standard_normal(sr * 3), burst(2.0), 0.001 * rng.standard_normal(int(sr * 0.55)), burst(2.0), 0.001 * rng.standard_normal(sr * 4)])
with wave.open('/tmp/mic.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(mic, -1, 1) * 32767).astype(np.int16).tobytes())
time.sleep(1.5)

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name, flush=True)

def status(url):
    try: return urllib.request.urlopen(url, timeout=3).status
    except urllib.error.HTTPError as e: return e.code
    except Exception: return 0

check('health without token is hidden (404)', status('http://127.0.0.1:8882/health') == 404)
check('health with wrong token is hidden (404)', status('http://127.0.0.1:8882/k/nope/health') == 404)
check('health with the token works', status(f'http://127.0.0.1:8882/k/{TOKEN}/health') == 200)
h = json.load(urllib.request.urlopen(f'http://127.0.0.1:8882/k/{TOKEN}/health'))
check('health reports ready with ears and voice flags', h.get('ready') is True and 'ears_ready' in h and 'voice_ready' in h)
check('brain model list passes through with the token', json.load(urllib.request.urlopen(f'http://127.0.0.1:8882/k/{TOKEN}/v1/models'))['data'][0]['id'] == 'fake-qwen')
import asyncio, websockets
async def ws_result(path):
    try:
        async with websockets.connect('ws://127.0.0.1:8882' + path, origin='http://localhost:8000') as w:
            await w.send(json.dumps({'type': 'config', 'lang': 'English'}))
            await asyncio.wait_for(w.recv(), 2)
            return 'open'
    except websockets.ConnectionClosed as e:
        return 'refused'
    except asyncio.TimeoutError:
        return 'open'
    except Exception:
        return 'refused'
check('listen socket refused with a wrong token', asyncio.run(ws_result('/k/nope/v1/listen')) == 'refused')
check('listen socket refused without a token', asyncio.run(ws_result('/v1/listen')) == 'refused')
check('listen socket accepted with the token', asyncio.run(ws_result(f'/k/{TOKEN}/v1/listen')) == 'open')
check('brain model list is hidden without the token', status('http://127.0.0.1:8882/v1/models') == 404)

BASE = f'http://127.0.0.1:8882/k/{TOKEN}/v1'
CORS = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'}
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-audio-capture=/tmp/mic.wav', '--autoplay-policy=no-user-gesture-required'])
    spy = "window.__tts = 0; if (window.speechSynthesis) { const o = window.speechSynthesis.speak.bind(window.speechSynthesis); window.speechSynthesis.speak = (u) => { window.__tts++; return o(u); }; }"
    seed = "localStorage.setItem('crane.settings', JSON.stringify({voiceV: 2, berylUrl: '%s', gpuBase: '%s', defaultModel: 'fake-qwen'}))" % (BASE, BASE)

    def page(extra_seed=seed):
        pg = b.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)[:160]))
        pg.add_init_script(spy); pg.add_init_script(extra_seed)
        pg.route('**/localhost:11435/**', lambda r: r.abort()); pg.route('**/127.0.0.1:888[01]/**', lambda r: r.abort())
        pg.route('**/router.huggingface.co/**', lambda r: r.fulfill(status=503, body='busy', headers=CORS))
        return pg, errs

    pg, errs = page()
    pg.goto('http://localhost:8000/app.html')
    pg.wait_for_selector('.msg.me', timeout=30000)
    pg.wait_for_selector('text=Let me start on that.', timeout=20000)
    pg.wait_for_timeout(1500)
    check('mic audio reached Qwen3-ASR through the token URL (English)', calls['transcribe'] and all(l == 'English' for _, l in calls['transcribe']))
    check('final transcript became a chat turn', 'build me a landing page' in pg.locator('.msg.me').first.inner_text())
    sysmsg = calls['brain'][0]['messages'][0]['content'] if calls['brain'] else ''
    check('brain request came through the worker pass-through', bool(calls['brain']) and calls['brain'][0]['model'] == 'fake-qwen')
    check('engineered spoken-style rules reached the brain', 'under eight words' in sysmsg and 'stage directions' in sysmsg)
    check('spoken turn uses conversation warmth 0.8', calls['brain'][0]['temperature'] == 0.8)
    check('greeting and reply spoken by Qwen3-TTS', len(calls['speak']) >= 2 and any('Let me start on that' in s for s in calls['speak']))
    check('browser speech synthesis never used', pg.evaluate('window.__tts') == 0)
    check('mid-sentence pause was answered with a backchannel clip from the worker', bool(calls['clips']))
    check('the clip was played in the browser, quietly', pg.evaluate('window.__craneTest.clipPlays') >= 1)
    check('the nod did not cut the turn in two: one brain request for the whole sentence', len(calls['brain']) == 1 and 'in gold' in pg.locator('.msg.me').first.inner_text())
    bd = lambda h, sp, bi: pg.evaluate('([h, s, b]) => window.__craneTest.bargeDecision(h, s, b)', [h, sp, bi])
    check('user says "mm-hm" while she talks: she keeps talking', bd('Mm-hm.', True, True) == 'ignore' and bd('yeah, right', True, True) == 'ignore' and bd('Oh wow', True, True) == 'ignore')
    check('user says something real while she talks: she stops', bd('wait, can you make it bigger', True, True) == 'interrupt' and bd('no', True, True) == 'interrupt')
    check('barge-in off: she never stops', bd('wait, stop', True, False) == 'ignore')
    check('she is not talking: everything is a normal turn', bd('mm-hm', False, True) == 'turn' and bd('build me a page', False, True) == 'turn')
    check('four-word yeah-yeah-yeah-yeah is a real sentence, not a nod', bd('yeah yeah yeah yeah', True, True) == 'interrupt')
    pg.click('.btn-admin'); pg.locator('.adm-nav-item', has_text='Voice Agent').click()
    check('admin shows the Beryl GPU server field', pg.locator('text=Beryl GPU server').count() == 1)
    check('no page errors', not errs)
    for e in errs: print('   ', e)

    pg2, errs2 = page("localStorage.setItem('crane.settings', JSON.stringify({voiceV: 2, berylUrl: 'http://127.0.0.1:8899/v1', gpuBase: 'http://127.0.0.1:8899/v1'}))")
    pg2.goto('http://localhost:8000/app.html'); pg2.wait_for_selector('.model-chip', timeout=30000); pg2.wait_for_timeout(4000)
    st = pg2.locator('#voiceStateLabel').get_attribute('title') or ''
    check('offline worker is reported (no Chrome fallback)', 'offline' in st)
    check('no page errors when offline', not errs2)

    pg.close(); pg2.close()
    pg3, _ = page(seed.replace(TOKEN, 'wrong-token'))
    n = len(calls['transcribe']); pg3.goto('http://localhost:8000/app.html'); pg3.wait_for_timeout(7000)
    check('wrong token in the browser: listening is refused', len(calls['transcribe']) == n)
    b.close()
web.kill()
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
