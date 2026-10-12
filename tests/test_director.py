"""Properties of the behavior Director (scripts/beryl_director.py). Pure Python.  Run:  python tests/test_director.py"""
import math, os, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from beryl_director import Director, rollout, affect_from_text, CHANNELS, FPS

fails = 0


def check(name, cond, extra=''):
    global fails
    fails += (not cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if (extra and not cond) else ''))


def col(frames, ch): return [f[ch] for f in frames]


def blinks(frames):
    """Counts blinks (closure crossing 0.5) and returns their durations in seconds above 0.5."""
    out, run = [], 0
    for f in frames:
        if f['blink'] > 0.5: run += 1
        elif run: out.append(run / FPS); run = 0
    return out


# --- blinking ---------------------------------------------------------------------------------------------
for state, lo, hi in (('listening', 9, 20), ('speaking', 11, 24), ('thinking', 7, 17)):
    d = Director(seed=4); d.event('state', state)
    fr = rollout(d, 300)
    per_min = len(blinks(fr)) / 5.0
    check(f'{state}: blink rate {lo}-{hi} per minute', lo <= per_min <= hi, round(per_min, 1))
fr = rollout((lambda d: (d.event('state', 'listening'), d)[1])(Director(seed=5)), 120)
dur = blinks(fr)
check('blinks last about 60-200 ms', dur and 0.04 <= statistics.mean(dur) <= 0.2, statistics.mean(dur))
check('each blink fully closes the eye', max(col(fr, 'blink')) >= 0.99)

# --- smoothness and bounds ----------------------------------------------------------------------------------
d = Director(seed=7); d.event('state', 'speaking')
fr = rollout(d, 120, {10: ('affect', ('amused', 0.8)), 20: ('nod', 0.8), 30: ('emphasis', None), 40: ('state', 'listening'), 70: ('state', 'thinking'), 90: ('state', 'speaking')})
limits = {'blink': 0.6, 'gaze_x': 0.5, 'gaze_y': 0.5, 'brow_raise': 0.35, 'brow_furrow': 0.1, 'smile': 0.1, 'head_yaw': 0.9, 'head_pitch': 1.4, 'head_roll': 0.6, 'breath': 0.12}
for ch in CHANNELS:
    x = col(fr, ch)
    worst = max(abs(b - a) for a, b in zip(x, x[1:]))
    check(f'{ch}: never jumps more than {limits[ch]} between frames', worst <= limits[ch], round(worst, 3))
bounds = {'blink': (0, 1), 'gaze_x': (-1, 1), 'gaze_y': (-1, 1), 'brow_raise': (0, 1), 'brow_furrow': (0, 1), 'smile': (0, 1), 'head_yaw': (-12, 12), 'head_pitch': (-14, 10), 'head_roll': (-10, 10), 'breath': (0, 1)}
for ch, (lo, hi) in bounds.items():
    x = col(fr, ch)
    check(f'{ch}: stays within [{lo}, {hi}]', min(x) >= lo - 1e-9 and max(x) <= hi + 1e-9, (round(min(x), 2), round(max(x), 2)))
idle = rollout(Director(seed=9), 120)
sd = statistics.pstdev(col(idle, 'head_yaw'))
check('idle: she is never frozen (head sway about 0.3-1.5 degrees)', 0.3 <= sd <= 1.5, round(sd, 2))

# --- eye contact ---------------------------------------------------------------------------------------------------
def contact(state, seed):
    d = Director(seed=seed); d.event('state', state)
    fr = rollout(d, 240)
    return sum(1 for f in fr if abs(f['gaze_x']) < 0.15 and abs(f['gaze_y']) < 0.15) / len(fr)
check('listening: mostly looks at the person (55-90%)', 0.55 <= statistics.mean(contact('listening', s) for s in range(5)) <= 0.9)
check('speaking: about half the time (35-70%)', 0.35 <= statistics.mean(contact('speaking', s) for s in range(5)) <= 0.7)
check('thinking: looks away most of the time (under 35% contact)', statistics.mean(contact('thinking', s) for s in range(5)) < 0.35)
d = Director(seed=3); d.event('state', 'thinking')
fr = rollout(d, 120)
check('thinking: eyes drift upward on average', statistics.mean(col(fr, 'gaze_y')) > 0.1, statistics.mean(col(fr, 'gaze_y')))

# --- nods ----------------------------------------------------------------------------------------------------------
d = Director(seed=1); d.event('state', 'listening')
base = rollout(d, 6)
fr = rollout(d, 3, {0.0: ('nod', 0.8)})
dip = min(col(fr, 'head_pitch')) - statistics.mean(col(base, 'head_pitch'))
check('backchannel nod dips the head at least 3 degrees', dip <= -3.0, round(dip, 2))
tail = rollout(d, 2)
check('...and settles back afterwards', abs(statistics.mean(col(tail, 'head_pitch'))) < 2.0)
check('a soft nod is smaller than a big one', min(col(rollout((lambda x: (x.event('nod', 0.3), x)[1])(Director(seed=1)), 2), 'head_pitch')) > dip)

# --- smile ---------------------------------------------------------------------------------------------------------
d = Director(seed=2); d.event('state', 'speaking')
rollout(d, 3)
fr = rollout(d, 8, {0.0: ('affect', ('amused', 0.8))})
sm = col(fr, 'smile')
rise = next(i for i, s in enumerate(sm) if s >= 0.5) / FPS
check('amusement: smile passes 0.5 within 0.8 s', rise <= 0.8, rise)
check('amusement: smile fades back near resting within 5 s of the end', sm[-1] < 0.25, round(sm[-1], 2))
check('amusement: the fade is slower than the rise', (len(sm) - 1 - max(i for i, s in enumerate(sm) if s >= 0.5)) / FPS > rise)
d = Director(seed=2); d.event('state', 'speaking')
fr = rollout(d, 4, {0.0: ('affect', ('concerned', 0.8))})
check('concern: brows furrow and she does not smile', max(col(fr, 'brow_furrow')) > 0.25 and max(col(fr, 'smile')) < 0.25)
d = Director(seed=2)
fr = rollout(d, 3, {0.0: ('emphasis', None)})
check('emphasis flashes the brows', max(col(fr, 'brow_raise')) > 0.3)

# --- speech energy drives head beats --------------------------------------------------------------------------------
def speak_run(pulsing):
    d = Director(seed=6); d.event('state', 'speaking')
    fr = []
    for i in range(10 * FPS):
        e = (0.9 if (i // 6) % 2 == 0 else 0.1) if pulsing else 0.0
        d.event('speech_energy', e)
        fr.append(d.step())
    return statistics.pstdev(col(fr, 'head_pitch'))
check('stressed syllables move the head (beats)', speak_run(True) > speak_run(False) * 1.4)

# --- breathing, determinism -------------------------------------------------------------------------------------------
x = col(rollout(Director(seed=8), 60), 'breath')
check('breath cycles slowly (about 0.2 Hz)', 8 <= sum(1 for a, b in zip(x, x[1:]) if a < 0.5 <= b) <= 16, sum(1 for a, b in zip(x, x[1:]) if a < 0.5 <= b))
a = rollout(Director(seed=21), 30); b = rollout(Director(seed=21), 30); c = rollout(Director(seed=22), 30)
check('same seed: identical behavior', a == b)
check('different seed: different behavior', a != c)

# --- reading a sentence ------------------------------------------------------------------------------------------------
cases = {"Ha, okay, that's actually hilarious.": 'amused', "I'm so sorry, that sounds rough.": 'concerned',
         "Oh wow, that's amazing!": 'excited', "Hm, what do you think about the layout?": 'curious',
         "Good morning, nice to see you.": 'warm', "I moved the button to the right.": 'neutral'}
for text, want in cases.items():
    got, inten = affect_from_text(text)
    check(f'reads "{text[:34]}..." as {want}', got == want and (inten > 0) == (want != 'neutral'), (got, inten))

print('ALL PASS' if not fails else f'{fails} FAILED'); sys.exit(1 if fails else 0)
