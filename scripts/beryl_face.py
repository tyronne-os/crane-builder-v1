"""Beryl's face: SoulX-FlashHead turns her speech audio into video frames of her portrait.

FlashHead renders in chunks: each call takes about one second of new audio (its `slice_len` frames) and returns that many
frames, carrying motion over from the previous chunk. That mirrors its own streaming loop (generate_video.py,
`audio_encode_mode == 'stream'`). Frames are timestamped from the start of the utterance; the browser shows each one when
the matching audio is playing, so lips stay in sync whatever speed the GPU renders at.

Wire format between processes and to the browser, one frame per record:
    4-byte big-endian JPEG length | 4-byte big-endian float32 seconds | JPEG bytes

Nothing here loads a model on import; FlashHeadRenderer needs the FlashHead checkout and its own Python environment.
"""
import io
import struct
from collections import deque

import numpy as np

FPS = 25


class Renderer:
    """Interface. chunk_samples: 16 kHz samples consumed per render() call. render(audio16k) -> uint8 frames (n, H, W, 3)."""
    chunk_samples = 640

    def render(self, audio16k):
        raise NotImplementedError


class FlashHeadRenderer(Renderer):
    """Real renderer (GPU). Mirrors FlashHead's stream mode call for call. Not run on a GPU from this repo yet."""

    def __init__(self, ckpt_dir, wav2vec_dir, cond_image, model_type="lite", base_seed=420):
        from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
        self._embed, self._run = get_audio_embedding, run_pipeline
        self.pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir, model_type=model_type)
        get_base_data(self.pipeline, cond_image_path_or_dir=cond_image, base_seed=base_seed, use_face_crop=True)
        p = get_infer_params()
        self.sample_rate, self.fps = p["sample_rate"], p["tgt_fps"]
        self.frame_num, self.motion = p["frame_num"], p["motion_frames_num"]
        slice_len = self.frame_num - self.motion
        cached = self.sample_rate * p["cached_audio_duration"]
        self.dq = deque([0.0] * cached, maxlen=cached)
        self.end_idx = p["cached_audio_duration"] * self.fps
        self.start_idx = self.end_idx - self.frame_num
        self.native_chunk = slice_len * self.sample_rate // self.fps
        self.chunk_samples = slice_len * 16000 // self.fps

    def render(self, audio16k):
        import torch
        audio = _resample(audio16k, 16000, self.sample_rate)[: self.native_chunk]
        if len(audio) < self.native_chunk:
            audio = np.concatenate([audio, np.zeros(self.native_chunk - len(audio), np.float32)])
        self.dq.extend(audio.tolist())
        with torch.no_grad():
            emb = self._embed(self.pipeline, np.array(self.dq), self.start_idx, self.end_idx)
            video = self._run(self.pipeline, emb)[self.motion:]
        return video.cpu().numpy().astype(np.uint8)


def _resample(x, src, dst):
    x = np.asarray(x, np.float32)
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def render_utterance(renderer, audio16k, fps=FPS):
    """Yields (seconds_from_utterance_start, frame) for one utterance, chunk by chunk, as soon as each chunk is rendered."""
    step = renderer.chunk_samples
    audio = np.asarray(audio16k, np.float32)
    if len(audio) % step:
        audio = np.concatenate([audio, np.zeros(step - len(audio) % step, np.float32)])
    n = 0
    for i in range(0, len(audio), step):
        for frame in renderer.render(audio[i:i + step]):
            yield n / fps, frame
            n += 1


def idle_loop(renderer, seconds=6.0, fps=FPS):
    """Silent breathing/blinking loop rendered once (the GPU rests while she is quiet). Cut where the end best matches
    the start so it repeats without a jump; the last few frames cross-fade into the first."""
    n = int(round(seconds * fps))
    frames = [f for _, f in render_utterance(renderer, np.zeros(int((seconds + 1.0) * 16000), np.float32), fps)]
    start = frames[0].astype(np.float32)
    lo, hi = max(1, n - fps), min(len(frames) - 1, n + fps)
    cut = min(range(lo, hi + 1), key=lambda i: float(np.mean(np.abs(frames[i].astype(np.float32) - start))))
    loop = frames[:cut]
    blend = min(3, len(loop) // 4)
    for i in range(blend):
        w = (i + 1) / (blend + 1)
        loop[-blend + i] = (loop[-blend + i].astype(np.float32) * (1 - w) + loop[i].astype(np.float32) * w).astype(np.uint8)
    return loop


def encode_jpeg(frame, quality=82):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pack(t, jpeg):
    return struct.pack(">If", len(jpeg), float(t)) + jpeg


def unpack_stream(buf):
    """Splits a byte buffer of packed records. Returns ([(t, jpeg)], leftover_bytes)."""
    out, i = [], 0
    while len(buf) - i >= 8:
        n, t = struct.unpack(">If", buf[i:i + 8])
        if len(buf) - i - 8 < n:
            break
        out.append((t, bytes(buf[i + 8:i + 8 + n])))
        i += 8 + n
    return out, buf[i:]
