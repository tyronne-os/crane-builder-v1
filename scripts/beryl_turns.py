"""Turn-taking for Beryl: when the user has finished, when to wait, and when to say "mm-hm".

Segmenter  : energy voice detection over a 16 kHz PCM stream. Emits speech start, partial snapshots, a "pause" probe
             240 ms into each silence, and the final utterance once the silence reaches the hold time the TurnTaker chose.
TurnTaker  : looks at the transcript so far and decides how long to hold the floor open (the "mini-turn" decision) and
             whether this pause is a good place for a backchannel.

Pure Python + numpy, no models, so it is unit-tested on its own (tests/test_turns.py).
"""
import random
import re

import numpy as np

SR = 16000
FRAME = SR // 50  # 20 ms
PROBE_MS = 240

# A sentence that ends on one of these is almost never finished: the speaker is mid-thought.
CONTINUERS = {
    "and", "but", "so", "because", "cause", "or", "then", "like", "um", "uh", "uhm", "er", "erm", "hmm", "well", "also",
    "plus", "since", "if", "when", "while", "that", "which", "who", "with", "to", "of", "the", "a", "an", "my", "your",
    "our", "their", "his", "her", "is", "are", "was", "were", "i", "it's", "i'm", "we", "you", "they", "for", "in", "on",
    "at", "about", "as", "than", "kind", "sort", "maybe", "actually", "basically", "just", "really", "very", "not",
}
QUESTION_STARTS = {"what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "do", "does",
                   "did", "is", "are", "will", "have", "has"}
BACKCHANNELS = ["mm-hm", "yeah", "right", "okay", "mm", "oh"]


class Segmenter:
    """Turns a PCM16-as-float32 stream into events: ("start",), ("partial", audio), ("pause", audio, quiet_ms), ("final", audio)."""

    def __init__(self, start_ms=60, hold_ms=700, preroll_ms=300, partial_ms=600, max_s=20):
        self.start_n, self.hold_n = start_ms // 20, hold_ms // 20
        self.default_hold_n = self.hold_n
        self.probe_n = PROBE_MS // 20
        self.preroll, self.partial_n, self.max_n = preroll_ms // 20, partial_ms // 20, max_s * 50
        self.floor, self.pending = 0.003, np.zeros(0, np.float32)
        self.ring, self.utt, self.talking, self.loud, self.quiet, self.since = [], [], False, 0, 0, 0

    def set_hold(self, ms):
        self.hold_n = max(self.probe_n, int(ms) // 20)

    @property
    def utterance_seconds(self):
        return len(self.utt) / 50.0

    def force_final(self):
        if not self.talking:
            return None
        audio = np.concatenate(self.utt)
        self.talking, self.utt, self.ring, self.loud, self.quiet = False, [], [], 0, 0
        return audio

    def feed(self, pcm):
        self.pending = np.concatenate([self.pending, pcm])
        while len(self.pending) >= FRAME:
            f, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            rms = float(np.sqrt(np.mean(f * f)))
            voiced = rms > max(self.floor * 3.0, 0.008)
            if not self.talking:
                self.floor = 0.95 * self.floor + 0.05 * min(rms, 0.05) if not voiced else self.floor
                self.ring = (self.ring + [f])[-self.preroll:]
                self.loud = self.loud + 1 if voiced else 0
                if self.loud >= self.start_n:
                    self.talking, self.utt, self.quiet, self.since = True, list(self.ring), 0, 0
                    self.hold_n = self.default_hold_n
                    yield ("start",)
                continue
            self.utt.append(f)
            self.quiet = 0 if voiced else self.quiet + 1
            self.since += 1
            if self.quiet >= self.hold_n or len(self.utt) >= self.max_n:
                audio = self.force_final()
                yield ("final", audio)
            elif self.quiet == self.probe_n:
                yield ("pause", np.concatenate(self.utt), self.quiet * 20)
            elif self.quiet == 0 and self.since >= self.partial_n:
                self.since = 0
                yield ("partial", np.concatenate(self.utt))


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class TurnTaker:
    """Decides, at each pause, how long to hold the floor and whether to backchannel. Deterministic given its seed."""

    def __init__(self, seed=None, backchannel_after_s=4.0, backchannel_gap_s=6.5, backchannel_chance=0.75):
        self.rng = random.Random(seed)
        self.after, self.gap, self.chance = backchannel_after_s, backchannel_gap_s, backchannel_chance
        self.reset()

    def reset(self):
        self.last_bc_at = -1e9
        self.last_kind = None
        self.clock = 0.0

    def hold_ms(self, text):
        """How much silence ends the turn. Short when the sentence is clearly finished, long when it is clearly not."""
        t = text.strip()
        w = _words(t)
        if not w:
            return 900
        tail = w[-1]
        trailing_off = t.endswith(("...", "…", ",", "-", "—", ":", ";"))
        if t.endswith("?"):
            return 300
        if trailing_off or tail in CONTINUERS:
            return 1500
        if t.endswith((".", "!")):
            return 380 if len(w) >= 4 else 600
        if len(w) < 3:
            return 900          # "hmm" / "so" ... they may be gathering a thought
        return 650              # ASR gave no punctuation: fall back to a normal pause

    def is_clause_boundary(self, text):
        t = text.strip()
        return t.endswith((",", ".", ";", "—", "-")) and not t.endswith("?")

    def decide(self, text, utterance_s, now_s):
        """Returns {"hold_ms": int, "backchannel": kind or None}."""
        hold = self.hold_ms(text)
        kind = None
        long_enough = utterance_s >= self.after and (now_s - self.last_bc_at) >= self.gap
        if long_enough and self.is_clause_boundary(text) and self.rng.random() < self.chance:
            options = [k for k in BACKCHANNELS[:4] if k != self.last_kind]
            kind = self.rng.choice(options)
            self.last_bc_at, self.last_kind = now_s, kind
            hold = max(hold, 900)   # we only nodded; the speaker has more to say
        return {"hold_ms": hold, "backchannel": kind}
