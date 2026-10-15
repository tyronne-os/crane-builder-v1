#!/usr/bin/env python3
"""Single source of truth for which models actually exist on this GPU box.

The Crane composer used to hardcode its model list in the browser, which meant the
UI showed rows for models that were never downloaded and stayed silent about ones
that were. This server reports what is really installed, so the composer can be a
view over reality instead of a guess.

It is a read-only aggregator. It never downloads, loads or evicts anything.

  GET /v1/models   OpenAI-shaped list. Each entry carries a "crane" block telling
                   the composer which backend to talk to and what modality it is.
  GET /health      per-backend reachability, for the composer's online indicator.

Backends, each optional and each tolerated being down:
  llama   /v1/models on 11434        (GGUF chat models, CUDA, the primary endpoint)
  cpu     /v1/models on 11437        (sub-1B GGUF, pure CPU, leaves VRAM to the 14B)
  ollama  native /api/tags on 11434  (legacy; only reported if it is what is running)
  vllm    /v1/models on 8000         (safetensors chat models, higher throughput)

Diffusion is deliberately NOT served here. Neither Ollama nor vLLM can generate
images or video, so a model appearing in this list means "chat only".
"""
import json
import os
import urllib.error
import urllib.request

LLAMA = os.environ.get("CRANE_LLAMA", "http://127.0.0.1:11434")
CPU = os.environ.get("CRANE_CPU", "http://127.0.0.1:11437")
OLLAMA = os.environ.get("CRANE_OLLAMA", "http://127.0.0.1:11434")
VLLM = os.environ.get("CRANE_VLLM", "http://127.0.0.1:8000")
LISTEN_PORT = int(os.environ.get("CRANE_REGISTRY_PORT", "11436"))
TIMEOUT = float(os.environ.get("CRANE_REGISTRY_TIMEOUT", "6"))

# Origins the hosted Crane Space is served from. Keep in sync with cors_proxy.py.
ALLOWED = (
    "https://aibruh-crane-builder-v1.static.hf.space",
    "https://aibruh-crane-builder-v1.hf.space",
    "https://huggingface.co",
)


def _get(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _gb(n):
    return None if not n else round(n / (1024 ** 3), 2)


def from_llama_cpu():
    """Sub-1B models served by a CPU-only llama-server. Kept distinct from the
    GPU list so the composer can label them and the user knows which card is free."""
    out = []
    for m in _get(CPU + "/v1/models").get("data", []):
        out.append({
            "id": m.get("id"),
            "object": "model",
            "owned_by": "cpu",
            "crane": {"source": "cpu", "backend": "llama-cpp", "modality": "chat", "device": "cpu"},
        })
    return out


def from_llama_gpu():
    out = []
    for m in _get(LLAMA + "/v1/models").get("data", []):
        out.append({
            "id": m.get("id"),
            "object": "model",
            "owned_by": "gpu",
            "crane": {"source": "gpu", "backend": "llama-cpp", "modality": "chat", "device": "cuda"},
        })
    return out


def from_ollama():
    """Native /api/tags carries size/family/quantisation that /v1/models drops."""
    out = []
    for m in _get(OLLAMA + "/api/tags").get("models", []):
        det = m.get("details") or {}
        out.append({
            "id": m.get("name") or m.get("model"),
            "object": "model",
            "owned_by": "gpu",
            "crane": {
                "source": "gpu",
                "backend": "ollama",
                "modality": "chat",
                "family": det.get("family"),
                "parameters": det.get("parameter_size"),
                "quantization": det.get("quantization_level"),
                "size_gb": _gb(m.get("size")),
            },
        })
    return out


def from_vllm():
    out = []
    for m in _get(VLLM + "/v1/models").get("data", []):
        out.append({
            "id": m.get("id"),
            "object": "model",
            "owned_by": "gpu",
            "crane": {
                "source": "gpu",
                "backend": "vllm",
                "modality": "chat",
            },
        })
    return out


BACKENDS = (("llama-gpu", from_llama_gpu), ("cpu", from_llama_cpu), ("ollama", from_ollama), ("vllm", from_vllm))


def collect():
    """Every reachable backend's models. A backend that is down is skipped, not fatal."""
    models, health = [], {}
    for name, fn in BACKENDS:
        try:
            got = fn()
            models.extend(got)
            health[name] = {"ok": True, "count": len(got)}
        except Exception as e:  # noqa: BLE001 - a down backend must not break the list
            health[name] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    return models, health


def main():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def cors(self):
            origin = self.headers.get("Origin", "")
            if origin in ALLOWED:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path.split("?")[0] in ("/health", "/v1/health"):
                _, health = collect()
                return self.reply(200, {"ok": any(v["ok"] for v in health.values()), "backends": health})
            models, _ = collect()
            self.reply(200, {"object": "list", "data": models})

        def log_message(self, fmt, *args):
            pass

    print("crane model registry: http://127.0.0.1:%d/v1/models (gpu=%s cpu=%s ollama=%s vllm=%s)"
          % (LISTEN_PORT, LLAMA, CPU, OLLAMA, VLLM), flush=True)
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
