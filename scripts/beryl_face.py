"""Beryl's face: lips from SoulX-FlashHead, everything else from the Behavior Director.

FaceStream turns the speech audio Beryl produces into video frames, chunk by chunk, the way FlashHead's own streaming
loop does (scripts/fh_generate.py, `audio_encode_mode == 'stream'`). Every frame is timestamped on the audio clock so the
browser can show it at the moment the matching sound leaves the speakers.

The Director's controls (blink, gaze, nods, brows, smile, head, breath) are passed to the renderer for every frame. The
renderer is an interface: FlashHeadRenderer runs the real model on a GPU; the tests use a fake renderer, so the timing,
pacing and control plumbing are checked without a GPU.

Nothing here runs on import (FlashHead needs its own environment: torch 2.7, flash-attn, transformers 4.57.3).
"""
import math
import os
import time
from collections import deque

import numpy as np

FPS = 25
FRAME_SAMPLES_16K = 640          # 40 ms of 16 kHz audio per FlashHead "tick" at 25 fps
MAX_LEAD_S = 0.6                 # never render more than this far ahead of the audio clock


class Renderer:
    """Interface. render(audio_window_16k, controls) -> uint8 array (n, H, W, 3)."""

    def render(self, audio_window, controls):
        raise NotImplementedError

    def still(self):
        raise NotImplementedError


class FlashHeadRenderer(Renderer):
    """Real renderer. Needs the FlashHead checkout (flash_head package) and its weights. Untested on a GPU here."""

    def __init__(self, ckpt_dir, wav2vec_dir, cond_image, model_type="lite", base_seed=420):
        from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
        self._get_audio_embedding, self._run_pipeline = get_audio_embedding, run_pipeline
        self.pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir, model_type=model_type)
        get_base_data(self.pipeline, cond_image_path_or_dir=cond_image, base_seed=base_seed, use_face_crop=True)
        p = get_infer_params()
        self.sample_rate, self.tgt_fps = p["sample_rate"], p["tgt_fps"]
        self.frame_num, self.motion_frames_num = p["frame_num"], p["motion_frames_num"]
        self.cached_audio_duration = p["cached_audio_duration"]
        self.slice_len = self.frame_num - self.motion_frames_num
        self.audio_dq = deque([0.0] * (self.sample_rate * self.cached_audio_duration), maxlen=self.sample_rate * self.cached_audio_duration)
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.first = True

    def render(self, audio_window, controls):
        import torch
        audio_24 = _resample(audio_window, 16000, self.sample_rate)
        self.audio_dq.extend(audio_24.tolist())
        emb = self._get_audio_embedding(self.pipeline, np.array(self.audio_dq), self.audio_start_idx, self.audio_end_idx)
        video = self._run_pipeline(self.pipeline, emb)
        if not self.first:
            video = video[self.motion_frames_num:]
        self.first = False
        frames = video.cpu().numpy().astype(np.uint8)
        return frames


def _resample(x, src, dst):
    if src == dst or len(x) == 0:
        return np.asarray(x, np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class FaceStream:
    """Feeds speech audio and Director controls in, yields timestamped frames out, paced to the audio clock.

    audio_clock: seconds of audio already played out to the listener (the browser reports it).
    """

    def __init__(self, renderer, director, fps=FPS):
        self.renderer, self.director, self.fps = renderer, director, fps
        self.audio = np.zeros(0, np.float32)     # 16 kHz speech not yet rendered
        self.t_render = 0.0                       # audio seconds already turned into frames
        self.frames_out = 0

    def push_audio(self, pcm16k):
        self.audio = np.concatenate([self.audio, np.asarray(pcm16k, np.float32)])

    def _controls_for_tick(self, tick_idx, audio_window):
        energy = float(np.sqrt(np.mean(audio_window ** 2))) if len(audio_window) else 0.0
        self.director.event("speech_energy", min(1.0, energy / 0.12))
        return self.director.step()

    def pump(self, audio_clock_s):
        """Renders as many ticks as are needed to stay about MAX_LEAD_S ahead of the audio clock.
        Returns a list of (frame_time_s, controls_dict, frame_uint8_array)."""
        out = []
        while len(self.audio) >= FRAME_SAMPLES_16K and self.t_render - audio_clock_s < MAX_LEAD_S:
            window = self.audio[:FRAME_SAMPLES_16K]
            self.audio = self.audio[FRAME_SAMPLES_16K:]
            ctrl = self._controls_for_tick(self.frames_out, window)
            frames = self.renderer.render(window, ctrl)
            n = len(frames)
            for i in range(n):
                t = self.t_render + i / self.fps
                # Each rendered frame's controls: the Director advances per frame, so the tick's first frame gets this ctrl.
                out.append((round(t, 4), ctrl if i == 0 else dict(ctrl), frames[i]))
            self.t_render += n / self.fps
            self.frames_out += n
        return out


def encode_jpeg(frame, quality=82):
    """Frames go to the browser as JPEG: about 15-25 KB at 512x512, cheap enough for 25 fps over a tunnel."""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def idle_loop(renderer, director, seconds=6.0):
    """A breathing-and-blinking loop rendered once, so the GPU is idle while she is silent.

    Renders a little more than `seconds`, then cuts at the frame whose picture best matches the start, so the loop repeats
    without a jump. A cross-fade is added only for the last few frames to hide any residual step."""
    n = int(round(seconds * FPS))
    silent = np.zeros(FRAME_SAMPLES_16K, np.float32)
    frames = []
    while len(frames) < n + FPS:
        frames.extend(renderer.render(silent, director.step()))
    start = frames[0].astype(np.float32)
    lo, hi = max(1, n - FPS), min(len(frames) - 1, n + FPS)
    cut = min(range(lo, hi + 1), key=lambda i: float(np.mean(np.abs(frames[i].astype(np.float32) - start))))
    loop = [f for f in frames[:cut]]
    blend = min(3, len(loop) // 4)
    for i in range(blend):
        w = (i + 1) / (blend + 1)
        loop[-blend + i] = (loop[-blend + i].astype(np.float32) * (1 - w) + loop[i].astype(np.float32) * w).astype(np.uint8)
    return loop
