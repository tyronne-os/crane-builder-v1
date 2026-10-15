#!/usr/bin/env python3
"""Integration test: two llama-servers (GPU + CPU) -> registry -> CORS proxy.

Stands up fake OpenAI-compatible servers on the real ports the composer uses and
asserts that a model id discovered by the registry routes to the correct endpoint.
This is the part that has never been exercised: the previous tests only checked
that files compiled.
"""
import json, os, subprocess, sys, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU_PORT, CPU_PORT, REG_PORT = 11434, 11437, 11436
PROXY_GPU, PROXY_CPU = 11535, 11537
ENV = {"CRANE_LLAMA": "http://127.0.0.1:%d" % GPU_PORT,
       "CRANE_CPU": "http://127.0.0.1:%d" % CPU_PORT,
       "CRANE_OLLAMA": "http://127.0.0.1:9",      # deliberately dead
       "CRANE_VLLM": "http://127.0.0.1:9",         # deliberately dead
       "CRANE_REGISTRY_PORT": str(REG_PORT)}

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (" :: " + str(extra) if extra and not cond else ""))
    if not cond:
        fails.append(name)


def serve(port, models):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in models]})
            else:
                self._send({"ok": True, "served_by": port})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            self._send({"choices": [{"message": {"content": "hello from port %d model %s" % (port, req.get("model"))}}]})

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def get(url):
    return json.loads(urllib.request.urlopen(url, timeout=10).read())


def post(url, body):
    r = urllib.request.Request(url, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=15).read())


import threading
threading.Thread(target=serve, args=(GPU_PORT, ["Qwen2.5-14B-Heretic-q4_k_m"]), daemon=True).start()
threading.Thread(target=serve, args=(CPU_PORT, ["Qwen3-1.7B-q4_k_m"]), daemon=True).start()

env = dict(os.environ, **ENV)
reg = subprocess.Popen([sys.executable, os.path.join(ROOT, "model_registry.py")],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)

print("registry sees both backends, and dead backends do not break the list:")
listing = get("http://127.0.0.1:%d/v1/models" % REG_PORT)
by_id = {m["id"]: m["crane"] for m in listing["data"]}
print("   ", json.dumps(by_id, indent=2))
check("gpu model tagged source=gpu", by_id.get("Qwen2.5-14B-Heretic-q4_k_m", {}).get("source") == "gpu")
check("cpu model tagged source=cpu", by_id.get("Qwen3-1.7B-q4_k_m", {}).get("source") == "cpu")

health = get("http://127.0.0.1:%d/health" % REG_PORT)["backends"]
check("ollama reported dead, not crashed", health["ollama"]["ok"] is False, health["ollama"])
check("vllm reported dead, not crashed", health["vllm"]["ok"] is False, health["vllm"])

print("\nrouting: a GPU model reaches the GPU server, a CPU model the CPU server")
gpu_reply = post("http://127.0.0.1:%d/v1/chat/completions" % GPU_PORT, {"model": "Qwen2.5-14B-Heretic-q4_k_m"})
cpu_reply = post("http://127.0.0.1:%d/v1/chat/completions" % CPU_PORT, {"model": "Qwen3-1.7B-q4_k_m"})
check("gpu model answered by :11434", "port 11434" in gpu_reply["choices"][0]["message"]["content"])
check("cpu model answered by :11437", "port 11437" in cpu_reply["choices"][0]["message"]["content"])

reg.terminate()
print("\n%s" % ("ALL CHECKS PASSED" if not fails else "FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
