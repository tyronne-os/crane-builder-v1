"""Face process: runs SoulX-FlashHead on the GPU and streams Beryl's frames to the worker (127.0.0.1:8883, never exposed).

Separate process because FlashHead pins its own torch/transformers versions. The worker (beryl_gpu_server.py) relays the
frames to the browser through its existing tunnel and token, so nothing new is opened to the internet.

GET  /health
GET  /idle     the cached silent loop, packed frames (see beryl_face.pack)
POST /render   body: 16 kHz mono PCM16 of one utterance -> packed frames, streamed as each ~1 s chunk is rendered

Run inside FlashHead's venv:  python beryl_face_server.py   (env: FLASHHEAD_DIR, BERYL_PORTRAIT, FLASHHEAD_MODEL=lite)
"""
import asyncio
import os
import threading

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from beryl_face import encode_jpeg, idle_loop, pack, render_utterance

PORT = int(os.environ.get("BERYL_FACE_PORT", "8883"))


def make_face_app(renderer):
    app = FastAPI()
    lock = threading.Lock()
    state = {"idle": None, "error": ""}

    def build_idle():
        try:
            with lock:
                frames = idle_loop(renderer)
            state["idle"] = b"".join(pack(i / 25.0, encode_jpeg(f)) for i, f in enumerate(frames))
        except Exception as e:  # reported by /health
            state["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=build_idle, daemon=True).start()

    @app.get("/health")
    def health():
        return {"ok": True, "ready": state["idle"] is not None, "error": state["error"]}

    @app.get("/idle")
    def idle():
        if state["idle"] is None:
            return Response(status_code=503)
        return Response(state["idle"], media_type="application/octet-stream")

    @app.post("/render")
    async def render(request: Request):
        pcm = np.frombuffer(await request.body(), dtype=np.int16).astype(np.float32) / 32768.0
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def work():
            try:
                with lock:  # one GPU, one utterance at a time, in order
                    for t, frame in render_utterance(renderer, pcm):
                        loop.call_soon_threadsafe(q.put_nowait, pack(t, encode_jpeg(frame)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=work, daemon=True).start()

        async def stream():
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        return StreamingResponse(stream(), media_type="application/octet-stream")

    return app


if __name__ == "__main__":
    import uvicorn
    from beryl_face import FlashHeadRenderer
    root = os.environ.get("FLASHHEAD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "SoulX-FlashHead"))
    renderer = FlashHeadRenderer(
        ckpt_dir=os.path.join(root, "models", "SoulX-FlashHead-1_3B"),
        wav2vec_dir=os.path.join(root, "models", "wav2vec2-base-960h"),
        cond_image=os.environ.get("BERYL_PORTRAIT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "beryl.jpeg")),
        model_type=os.environ.get("FLASHHEAD_MODEL", "lite"))
    print(f"[beryl-face] http://127.0.0.1:{PORT}  rendering the idle loop now; /health says ready when done")
    uvicorn.run(make_face_app(renderer), host="127.0.0.1", port=PORT, log_level="warning")
