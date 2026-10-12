"""Generate Beryl's voice with the real Qwen3-TTS models and measure it objectively.

Designs the voice (Qwen3-TTS-VoiceDesign), clones it for four different lines (Qwen3-TTS-Base), then transcribes each
clip with Qwen3-ASR to check it is intelligible, and measures pitch, pace, level and clipping. Works on CPU (slow, same
weights) or GPU.  Usage:  BERYL_DEVICE=cpu python tests/voice_verify.py OUT_DIR
"""
import json, os, re, sys, time
import numpy as np, soundfile as sf, librosa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/voice_out'
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault('BERYL_VOICE_DIR', os.path.join(OUT, 'voices'))
import beryl_gpu_server as B

LINES = {
    'greeting': "Hey, there you are. I was just thinking about what we could build today.",
    'excited': "Oh, wait, that's actually really good. Okay, so, we could put the headline right in the middle, and let it breathe a little.",
    'caring': "Hm, yeah, that sounds rough. I'm sorry. Do you want to talk about it, or should I just stay here with you for a bit?",
    'working': "Okay, I changed the button so it opens the builder. Give it a click and tell me what you think.",
}


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0], d[0, :] = range(len(r) + 1), range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def measure(wav, sr):
    y = wav.astype(np.float32)
    f0, voiced, _ = librosa.pyin(librosa.resample(y, orig_sr=sr, target_sr=16000), fmin=70, fmax=400, sr=16000)
    f0 = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    active = np.where(rms > 0.02 * rms.max())[0]
    return {
        'seconds': round(len(y) / sr, 2),
        'lead_silence_s': round(active[0] * 512 / sr, 2) if len(active) else None,
        'trail_silence_s': round((len(y) - (active[-1] + 1) * 512) / sr, 2) if len(active) else None,
        'pitch_median_hz': round(float(np.median(f0)), 1) if len(f0) else None,
        'pitch_spread_semitones': round(float(np.std(12 * np.log2(f0 / np.median(f0)))), 2) if len(f0) else None,
        'level_dbfs': round(20 * np.log10(float(np.sqrt(np.mean(y ** 2))) + 1e-9), 1),
        'peak': round(float(np.max(np.abs(y))), 3),
        'clipped_fraction': round(float(np.mean(np.abs(y) > 0.999)), 5),
    }


m = B.Models()
report = {'device': B.DEVICE, 'tts_model': B.TTS_MODEL, 'asr_model': B.ASR_MODEL, 'voice_prompt': B.VOICE_PROMPT, 'lines': {}}
for name, text in LINES.items():
    t0 = time.time()
    wav, sr = m.speak(text)
    gen = time.time() - t0
    wav = np.asarray(wav, dtype=np.float32)
    sf.write(os.path.join(OUT, f'beryl_{name}.wav'), wav, sr, subtype='PCM_16')
    heard = m.transcribe(librosa.resample(wav, orig_sr=sr, target_sr=16000), 'English')
    stats = measure(wav, sr)
    stats.update(text=text, heard=heard, word_error_rate=round(wer(text, heard), 3),
                 words_per_second=round(len(words(heard)) / max(0.1, stats['seconds'] - (stats['lead_silence_s'] or 0) - (stats['trail_silence_s'] or 0)), 2),
                 generation_seconds=round(gen, 1), real_time_factor=round(gen / stats['seconds'], 2))
    report['lines'][name] = stats
    print(name, json.dumps(stats), flush=True)
    json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
ref = [f for f in os.listdir(B.VOICE_DIR) if f.endswith('.wav')]
if ref:
    import shutil; shutil.copy(os.path.join(B.VOICE_DIR, ref[0]), os.path.join(OUT, 'beryl_designed_reference.wav'))
print('DONE', flush=True)
