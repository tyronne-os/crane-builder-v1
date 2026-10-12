"""Unit tests for scripts/beryl_turns.py (no models, no GPU). Run:  python tests/test_turns.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from beryl_turns import Segmenter, TurnTaker, BACKCHANNELS

rng = np.random.default_rng(1)
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


def run(seg, audio, chunk=1600):
    """Feeds audio in 100 ms chunks and returns [(event, seconds_into_stream)]."""
    out = []
    for i in range(0, len(audio), chunk):
        for ev in seg.feed(audio[i:i + chunk]):
            out.append((ev[0], round((i + chunk) / 16000, 2), ev))
    return out


# --- Segmenter -------------------------------------------------------------------------------------------------
seg = Segmenter()
ev = run(seg, np.concatenate([quiet(1), speech(2), quiet(2)]))
kinds = [e[0] for e in ev]
check('speech start is announced', 'start' in kinds)
check('exactly one pause probe per pause', kinds.count('pause') == 1, kinds)
check('turn ends with one final', kinds.count('final') == 1, kinds)
pause_t = next(t for k, t, _ in ev if k == 'pause'); final_t = next(t for k, t, _ in ev if k == 'final')
check('default hold: final about 700 ms after speech stops', 0.6 <= final_t - 3.0 <= 0.9, f'final at {final_t}')
check('probe fires about 240 ms into the pause', 0.2 <= pause_t - 3.0 <= 0.45, f'pause at {pause_t}')

seg = Segmenter()
events = []
for part in (np.concatenate([quiet(0.5), speech(1.5)]), quiet(0.3)):
    events += run(seg, part)
seg.set_hold(1500)
events += run(seg, quiet(0.9))
check('a longer hold keeps the turn open through a 1.2 s pause', 'final' not in [e[0] for e in events])
events += run(seg, quiet(0.5))
check('...and still ends it after the hold passes', 'final' in [e[0] for e in events])

seg = Segmenter()
run(seg, np.concatenate([quiet(0.5), speech(1.0)]))
audio = seg.force_final()
check('force_final takes the turn immediately', audio is not None and len(audio) > 16000)
check('force_final when nobody is talking returns nothing', seg.force_final() is None)

# --- TurnTaker: how long to hold ----------------------------------------------------------------------------------
tt = TurnTaker(seed=0)
check('question: short hold', tt.hold_ms('What do you think about the layout?') <= 350)
check('finished statement: short hold', tt.hold_ms('I want to change the headline.') <= 450)
check('ends on "and": wait a long time', tt.hold_ms('I want the button to be gold and') >= 1400)
check('ends on a comma: wait a long time', tt.hold_ms('So the first thing is,') >= 1400)
check('trailing off with ellipsis: wait', tt.hold_ms('well I was thinking...') >= 1400)
check('"hmm": give them room to think', tt.hold_ms('hmm') >= 800)
check('no punctuation from ASR: normal pause', 500 <= tt.hold_ms('make the header bigger on mobile') <= 800)
check('nothing heard yet: patient', tt.hold_ms('') >= 800)

# --- TurnTaker: backchannels --------------------------------------------------------------------------------------
tt = TurnTaker(seed=3, backchannel_chance=1.0)
d = tt.decide('So I started with the landing page, and then I moved the headline up,', 6.0, 10.0)
check('long turn at a clause boundary gets a backchannel', d['backchannel'] in BACKCHANNELS)
check('backchannel is not the "oh" reserved for surprise', d['backchannel'] != 'oh')
d2 = tt.decide('and then the button was too small,', 7.0, 12.0)
check('no second backchannel inside the gap', d2['backchannel'] is None)
d3 = tt.decide('and I fixed that, and now it is better,', 9.0, 20.0)
check('later pause can get another', d3['backchannel'] is not None and d3['backchannel'] != d['backchannel'])
tt2 = TurnTaker(seed=3, backchannel_chance=1.0)
check('short turns never get one', tt2.decide('I like it,', 1.5, 3.0)['backchannel'] is None)
check('never nods in the middle of a question', tt2.decide('what do you think about', 8.0, 30.0)['backchannel'] is None)
check('never after a question mark', tt2.decide('do you like it?', 8.0, 31.0)['backchannel'] is None)
check('backchannel extends the hold so she does not take the turn', TurnTaker(seed=3, backchannel_chance=1.0).decide('I did this, and', 6.0, 9.0)['hold_ms'] >= 900)

# Natural variety over many long turns: neither constant nor absent.
tt3 = TurnTaker(seed=11)
fired, t = 0, 0.0
for _ in range(200):
    t += 8.0
    fired += tt3.decide('and that is the part I like,', 7.0, t)['backchannel'] is not None
check('with default settings roughly 60-90% of eligible pauses get one', 120 <= fired <= 190, fired)

print('ALL PASS' if not fails else f'{fails} FAILED'); sys.exit(1 if fails else 0)
