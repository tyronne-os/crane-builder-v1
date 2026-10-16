#!/usr/bin/env python3
"""Crane terminal server — a real persistent WSL bash session on :11443.

One long-running `bash` process holds the session: cd, exports, venvs, git
state — everything persists across commands. Each POST /exec writes the
command plus a unique sentinel echo; the sentinel returning (with the exit
code) marks the command complete and slices its exact output.

CORS is open to the hosted builder (same pattern as the voice server).
No PTY: interactive programs (vim, htop) won't work — commands do, which
is the agent's bread and butter.
"""
import json
import os
import select
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 11443
bash = subprocess.Popen(
    ["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, bufsize=0, cwd=os.path.expanduser("~"),
    env=dict(os.environ, TERM="dumb", PS1=""))
fd = bash.stdout.fileno()

buf = ""
lock = threading.Lock()
served = 0  # how much of buf has been handed to a caller

def drain():
    global buf
    while True:
        r, _, _ = select.select([fd], [], [], 0.15)
        if r:
            try:
                data = os.read(fd, 65536)
                if data:
                    with lock:
                        buf += data.decode("utf-8", "ignore")
            except Exception:
                pass

threading.Thread(target=drain, daemon=True).start()

_mid = [0]

def exec_cmd(cmd, timeout):
    """Run one command; return (output, exit_code, completed)."""
    global served
    with lock:
        _mid[0] += 1
        marker = "__CRANE_%d__" % _mid[0]
        # clear anything unserved from background output first
        pre = buf[served:]
        served = len(buf)
        bash.stdin.write((cmd + "\necho " + marker + "$?\n").encode())
        bash.stdin.flush()
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.12)
        with lock:
            idx = buf.find(marker, served)
            if idx >= 0:
                out = pre + buf[served:idx]
                exit_code = buf[idx + len(marker):].split("\n", 1)[0].strip()
                served = idx + len(marker) + len(exit_code) + 1
                # clean control chars from non-tty bash
                out = out.replace("\r", "")
                return out.strip()[:20000], exit_code, True
    with lock:
        out = pre + buf[served:]
        served = len(buf)
    return out.strip()[:20000], "", False

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            return self._json({"status": "ok", "alive": bash.poll() is None, "cwd": os.getcwd()})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            d = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._json({"error": "bad json"}, 400)
        if self.path == "/exec":
            cmd = str(d.get("cmd", ""))[:8000]
            timeout = min(120, float(d.get("timeout", 20)))
            if not cmd.strip():
                return self._json({"error": "empty command"}, 400)
            out, code, done = exec_cmd(cmd, timeout)
            return self._json({"output": out, "exit": code, "completed": done})
        if self.path == "/poll":
            global served
            with lock:
                out = buf[served:]
                served = len(buf)
            return self._json({"output": out.replace("\r", "").strip()[:20000]})
        self._json({"error": "not found"}, 404)

if __name__ == "__main__":
    print("Crane terminal server on :11443 — a real persistent WSL bash", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
