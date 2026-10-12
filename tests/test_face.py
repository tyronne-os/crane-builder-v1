"""Face stream plumbing with a fake renderer (no GPU, no FlashHead). Run:  python tests/test_face.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from beryl_face import FaceStream, Renderer, FRAME_SAMPLES_16K, MAX_LEAD_S, FPS, idle_loop, encode_jpeg
from beryl_director import Director, CHANNELS

fails = 0


def check(name, cond, extra=''):
    global fails
    fails += (not cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if (extra and not cond) else ''))


class FakeRenderer(Renderer):
    """Each 40 ms tick yields 1 frame (FPS=25 means 1 frame per tick). Frame value encodes its tick index."""
    def __init__(self): self.calls, self.ctrls = 0, []
    def render(self, audio_window, controls):
        self.calls += 1
        self.ctrls.append(controls)
        f = np.full((4, 4, 3), self.calls % 256, np.uint8)
        return np.stack([f])
    def still(self): return np.zeros((1, 4, 4, 3), np.uint8)


def speech(sec):
    n = int(16000 * sec)
    return (0.2 * np.sin(np.linspace(0, 400 * sec, n))).astype(np.float32)


# --- pacing to the audio clock -----------------------------------------------------------------------------------
r = FakeRenderer(); fs = FaceStream(r, Director(seed=1))
fs.push_audio(speech(2.0))
out = fs.pump(audio_clock_s=0.0)
check('renders at most MAX_LEAD_S ahead of the audio clock', fs.t_render - 0.0 <= MAX_LEAD_S + 1 / FPS + 1e-9, round(fs.t_render, 3))
check('first pump produces frames', len(out) > 0)
check('frame times run at 25 fps', all(abs((out[i + 1][0] - out[i][0]) - 1 / FPS) < 1e-3 for i in range(len(out) - 1)))

out2 = fs.pump(audio_clock_s=0.0)
check('no more frames when already ahead by the lead', len(out2) == 0)
out3 = fs.pump(audio_clock_s=1.0)
check('playback progress releases more frames', len(out3) > 0 and out3[0][0] > out[-1][0] - 1e-9)

# --- starvation: no audio, no frames; audio arrives later, rendering resumes -----------------------------------
r2 = FakeRenderer(); fs2 = FaceStream(r2, Director(seed=2))
check('no audio yields no frames', fs2.pump(0.0) == [])
fs2.push_audio(np.zeros(FRAME_SAMPLES_16K // 2, np.float32))
check('a half tick is held until it is complete', fs2.pump(0.0) == [] and len(fs2.audio) > 0)
fs2.push_audio(speech(0.5))
check('once a tick is complete it renders', len(fs2.pump(0.0)) > 0)

# --- controls reach every frame, bounded and varying -----------------------------------------------------------
r3 = FakeRenderer(); d3 = Director(seed=3); fs3 = FaceStream(r3, d3)
fs3.push_audio(speech(6.0))
frames = []
clock = 0.0
for _ in range(400):
    frames += fs3.pump(clock)
    clock += 1 / FPS
    if len(frames) > 120: break
ctrls = [f[1] for f in frames]
check('every frame carries a full set of face controls', all(set(c) == set(CHANNELS) for c in ctrls))
check('face controls vary over time (the face is alive)', len({round(c['head_yaw'], 4) for c in ctrls}) > 20)
check('controls stay within their ranges', all(0 <= c['smile'] <= 1 and 0 <= c['blink'] <= 1 for c in ctrls))

# speech energy reaches the director so mouth-speaking beats move the head
loud = FaceStream(FakeRenderer(), Director(seed=4))
loud.push_audio(np.concatenate([speech(1.0), np.zeros(16000, np.float32)]))
quiet = FaceStream(FakeRenderer(), Director(seed=4))
quiet.push_audio(np.zeros(32000, np.float32))
lf = loud.pump(0.0); qf = quiet.pump(0.0)
check('energetic speech drives the Director (speech_energy events arrive)', loud.director.prev_energy > 0.5 or len(lf) > 0)

# --- frame data ----------------------------------------------------------------------------------------------------
check('frames are uint8 arrays', all(f[2].dtype == np.uint8 for f in frames))
jpg = encode_jpeg(np.zeros((64, 64, 3), np.uint8))
check('frames encode to JPEG', jpg[:2] == b'\xff\xd8')

# --- idle loop: rendered once, seamless --------------------------------------------------------------------------------
idle = idle_loop(FakeRenderer(), Director(seed=5), seconds=6.0)
check('idle loop is about 6 seconds long', abs(len(idle) - 6 * FPS) <= FPS, len(idle))
check('idle loop frames are uint8', all(f.dtype == np.uint8 for f in idle))
class Smooth(Renderer):
    def __init__(self): self.k = 0
    def render(self, audio_window, controls):
        self.k += 1
        v = 120 + 40 * np.sin(2 * np.pi * self.k / 60)
        return np.full((1, 4, 4, 3), int(v), np.uint8)
    def still(self): return np.zeros((1, 4, 4, 3), np.uint8)
loop = idle_loop(Smooth(), Director(seed=5), seconds=6.0)
seq = np.array([f.mean() for f in loop] + [loop[0].mean()])
steps = np.abs(np.diff(seq))
check('idle loop joins smoothly: the seam step stays within 2x the typical frame-to-frame step', steps[-1] <= 2 * np.percentile(steps, 95) + 1.0, (round(steps[-1], 2), round(float(np.percentile(steps, 95)), 2)))

print('ALL PASS' if not fails else f'{fails} FAILED'); sys.exit(1 if fails else 0)
