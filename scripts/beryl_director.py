"""Behavior Director: turns conversation events into smooth per-frame face controls (blink, gaze, brows, smile, head, breath).

The talking-head model (FlashHead) only knows how to move the lips for speech. The Director supplies everything a person
does around speech: blinking, eye contact and glances away while thinking, nods when you talk, a brow flash on emphasis,
a smile that rises and fades with what is being said, slow breathing and head sway. A second model (a LivePortrait-style
expression retargeter) applies these controls to the rendered face.

Pure Python, deterministic for a given seed, so every property is unit-tested (tests/test_director.py).

    d = Director(seed=1)
    d.event("state", "listening")
    d.event("nod", 0.8)
    d.event("affect", ("amused", 0.7))
    controls = d.step()          # advances 1/fps seconds; returns {channel: value}
"""
import math
import random
import re

FPS = 25
CHANNELS = ("blink", "gaze_x", "gaze_y", "brow_raise", "brow_furrow", "smile", "head_yaw", "head_pitch", "head_roll", "breath")

# blink_mean: seconds between blinks. contact: share of fixations on the camera. base_smile: resting smile.
STATES = {
    "idle":      dict(blink_mean=4.4, contact=0.40, base_smile=0.08, sway=1.0),
    "listening": dict(blink_mean=4.0, contact=0.62, base_smile=0.16, sway=0.8),
    "thinking":  dict(blink_mean=5.0, contact=0.12, base_smile=0.05, sway=0.7),
    "speaking":  dict(blink_mean=3.4, contact=0.32, base_smile=0.12, sway=1.2),
}
# What each feeling does to the face at full intensity.
AFFECTS = {
    "neutral":   dict(smile=0.0, raise_=0.0, furrow=0.0, roll=0.0),
    "warm":      dict(smile=0.40, raise_=0.10, furrow=0.0, roll=1.5),
    "amused":    dict(smile=0.78, raise_=0.20, furrow=0.0, roll=2.0),
    "excited":   dict(smile=0.65, raise_=0.45, furrow=0.0, roll=0.0),
    "concerned": dict(smile=0.0, raise_=0.22, furrow=0.45, roll=3.0),
    "curious":   dict(smile=0.18, raise_=0.35, furrow=0.0, roll=3.5),
}
AWAY = [(-0.55, 0.30), (0.55, 0.30), (-0.65, 0.0), (0.65, 0.0), (-0.30, -0.30), (0.35, -0.25)]
UP = [(-0.5, 0.45), (0.5, 0.45), (-0.25, 0.55)]


def affect_from_text(text):
    """Cheap, local read of how a sentence should look. Returns (affect, intensity)."""
    t = text.strip().lower()
    if re.search(r"\b(sorry|rough|tough|that sounds (hard|awful|terrible)|i'm here|hang in there|that's hard)\b", t):
        return "concerned", 0.8
    if re.search(r"\b(ha+|haha|hehe|lol|funny|hilarious|joke|kidding)\b", t):
        return "amused", 0.8
    if re.search(r"\b(wow|amazing|love (it|this|that)|awesome|incredible|oh my|no way|so good|excited)\b", t) or t.count("!") >= 2:
        return "excited", 0.8
    if t.endswith("?") or re.match(r"^(hm+|huh|wait|so,? what|what if|how come)\b", t):
        return "curious", 0.6
    if re.search(r"\b(hey|hi|hello|good (morning|afternoon|evening)|thanks|thank you|glad|happy|nice to)\b", t):
        return "warm", 0.7
    return "neutral", 0.0


class Director:
    def __init__(self, seed=None, fps=FPS):
        self.rng = random.Random(seed)
        self.fps, self.dt = fps, 1.0 / fps
        self.t = 0.0
        self.state = "idle"
        self.v = {c: 0.0 for c in CHANNELS}
        self.v["smile"] = STATES["idle"]["base_smile"]
        self.v["breath"] = 0.5
        # blink
        self.blink_start, self.blink_next, self.blink_double = None, self._interval(), False
        # gaze
        self.gaze_target, self.gaze_until = (0.0, 0.0), 0.0
        self._pick_gaze()
        # head sway: three slow sines per axis with random frequency/phase
        self.sway = {a: [(self.rng.uniform(0.07, 0.42), self.rng.uniform(0, 6.283), self.rng.uniform(0.4, 1.0)) for _ in range(3)] for a in ("yaw", "pitch", "roll")}
        self.nods = []        # (start, dur, amp, cycles)
        self.beats = []       # (start, amp)
        self.affect = ("neutral", 0.0, 0.0)  # name, intensity, hold_until
        self.brow_flash = []  # (start, amp)
        self.breath_pulse = 0.0
        self.prev_energy, self.beat_cooldown = 0.0, 0.0
        self.head_follow, self.roll_bias = 0.0, 0.0

    # ---- events ------------------------------------------------------------------------------------------
    def event(self, kind, arg=None):
        if kind == "state" and arg in STATES and arg != self.state:
            if arg == "speaking":
                self.breath_pulse = 1.0
                self.blink_next = min(self.blink_next, self.t + self.rng.uniform(0.15, 0.6))
            self.state = arg
            self.gaze_until = self.t     # re-pick gaze for the new state
        elif kind == "nod":
            amp = max(0.2, min(1.0, float(arg if arg is not None else 0.7)))
            self.nods.append((self.t, 0.95, 3.0 + 3.0 * amp, 2 if amp < 0.9 else 3))
        elif kind == "affect":
            name, inten = arg
            self.affect = (name if name in AFFECTS else "neutral", max(0.0, min(1.0, inten)), self.t + 2.4)
        elif kind == "emphasis":
            self.brow_flash.append((self.t, 0.5))
            self.beats.append((self.t, 2.2))
        elif kind == "sentence_end":
            self.gaze_until = self.t      # glance back to the person at the end of a thought
            self.gaze_force_contact = True
        elif kind == "speech_energy":
            e = float(arg)
            if e - self.prev_energy > 0.25 and self.t >= self.beat_cooldown and self.state == "speaking":
                self.beats.append((self.t, 1.0 + 1.6 * e))
                self.beat_cooldown = self.t + 0.38
                if e > 0.7 and self.rng.random() < 0.35:
                    self.brow_flash.append((self.t, 0.35))
            self.prev_energy = e

    # ---- internals ---------------------------------------------------------------------------------------
    def _interval(self):
        mean = STATES[getattr(self, "state", "idle")]["blink_mean"]
        return max(1.2, self.rng.lognormvariate(math.log(mean) - 0.08, 0.4))

    def _pick_gaze(self):
        p = STATES[self.state]["contact"] if hasattr(self, "state") else 0.5
        if getattr(self, "gaze_force_contact", False):
            p, self.gaze_force_contact = 1.0, False
        if self.rng.random() < p:
            self.gaze_target = (self.rng.uniform(-0.03, 0.03), self.rng.uniform(-0.03, 0.03))
            self.gaze_until = self.t + self.rng.uniform(1.4, 3.8)
        else:
            pool = UP if self.state == "thinking" else AWAY
            near = [g for g in pool if abs(g[0] - self.gaze_target[0]) <= 0.75] or pool
            self.gaze_target = self.rng.choice(near)
            self.gaze_until = self.t + self.rng.uniform(0.5, 1.5)

    def _blink(self):
        if self.blink_start is None:
            if self.t >= self.blink_next:
                self.blink_start = self.t
            return 0.0
        tau = self.t - self.blink_start
        if tau < 0.08:
            return tau / 0.08
        if tau < 0.11:
            return 1.0
        if tau < 0.23:
            return 1.0 - (tau - 0.11) / 0.12
        self.blink_start = None
        self.blink_next = self.t + (0.2 if self.rng.random() < 0.10 else self._interval())
        return 0.0

    def _head(self, axis, gain):
        return gain * sum(a * math.sin(2 * math.pi * f * self.t + ph) for f, ph, a in self.sway[axis])

    @staticmethod
    def _lag(cur, target, tau, dt):
        return cur + (target - cur) * (1.0 - math.exp(-dt / tau))

    def step(self):
        self.t += self.dt
        dt, t, st, v = self.dt, self.t, STATES[self.state], self.v
        if t >= self.gaze_until:
            self._pick_gaze()
        name, inten, hold_until = self.affect
        live = inten if t < hold_until else 0.0
        a = AFFECTS[name]
        # smile rises fast, falls slowly, like a real one
        smile_target = st["base_smile"] + a["smile"] * live
        v["smile"] = self._lag(v["smile"], smile_target, 0.30 if smile_target > v["smile"] else 1.1, dt)
        brow_target = a["raise_"] * live + sum(amp * math.sin(math.pi * min(1.0, (t - s) / 0.45)) for s, amp in self.brow_flash if 0 <= t - s <= 0.45)
        self.brow_flash = [(s, amp) for s, amp in self.brow_flash if t - s <= 0.45]
        v["brow_raise"] = self._lag(v["brow_raise"], min(1.0, brow_target), 0.10, dt)
        v["brow_furrow"] = self._lag(v["brow_furrow"], a["furrow"] * live, 0.35, dt)
        # gaze: fast saccade, slight fixation tremor
        tremor = 0.012 * math.sin(7.3 * t) + 0.008 * math.sin(13.1 * t + 1.0)
        v["gaze_x"] = self._lag(v["gaze_x"], self.gaze_target[0] + tremor, 0.06, dt)
        v["gaze_y"] = self._lag(v["gaze_y"], self.gaze_target[1] - tremor * 0.6, 0.06, dt)
        # head: slow sway + nods + beats on speech energy
        pitch = self._head("pitch", 0.45 * st["sway"])
        for s, dur, amp, cycles in self.nods:
            tau = t - s
            if 0 <= tau <= dur:
                pitch -= amp * math.sin(math.pi * cycles * tau / dur) ** 2 * math.sin(math.pi * tau / dur)
        self.nods = [n for n in self.nods if t - n[0] <= n[1]]
        for s, amp in self.beats:
            tau = t - s
            if 0 <= tau <= 0.35:
                pitch -= amp * math.sin(math.pi * tau / 0.35)
        self.beats = [b for b in self.beats if t - b[0] <= 0.35]
        v["head_pitch"] = pitch
        self.head_follow = self._lag(self.head_follow, 3.0 * self.gaze_target[0], 0.45, dt)
        self.roll_bias = self._lag(self.roll_bias, a["roll"] * live, 0.5, dt)
        v["head_yaw"] = self._head("yaw", 1.1 * st["sway"]) + self.head_follow
        v["head_roll"] = self._head("roll", 0.6 * st["sway"]) + self.roll_bias
        v["blink"] = self._blink()
        self.breath_pulse = max(0.0, self.breath_pulse - dt / 0.8)
        v["breath"] = self._lag(v["breath"], min(1.0, 0.5 + 0.4 * math.sin(2 * math.pi * 0.22 * t) + 0.3 * self.breath_pulse), 0.12, dt)
        return dict(v)


def rollout(director, seconds, schedule=None):
    """Runs a director for N seconds, applying {second: (kind, arg)} events; returns one control dict per frame."""
    frames, events = [], sorted((schedule or {}).items())
    for i in range(int(seconds * director.fps)):
        while events and events[0][0] <= i / director.fps:
            director.event(*events.pop(0)[1])
        frames.append(director.step())
    return frames
