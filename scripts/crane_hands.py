"""Crane Hands: local computer-use agent for Crane Builder (run with Windows Python).

Listens on 127.0.0.1:8765 only, requires the token in %USERPROFILE%\\.crane\\token.txt,
and only answers pages served from the Crane origins below.
Move the mouse into any screen corner to abort a running action (pyautogui failsafe).
"""
import base64, io, json, os, secrets, subprocess, sys, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

PORT = 8765
ALLOWED_ORIGINS = {
    "https://aibruh-crane-builder-v1.static.hf.space",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
}
MAX_ACTIONS = 50

TOKEN_FILE = Path.home() / ".crane" / "token.txt"
TOKEN_FILE.parent.mkdir(exist_ok=True)
if not TOKEN_FILE.exists():
    TOKEN_FILE.write_text(secrets.token_urlsafe(24))
TOKEN = TOKEN_FILE.read_text().strip()


def screenshot_b64(max_w=1280):
    img = pyautogui.screenshot()
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


CHROME_PATHS = [
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]


def do_open(target):
    if target.startswith(("http://", "https://")):
        chrome = next((p for p in CHROME_PATHS if os.path.isfile(p)), None)
        if chrome:
            subprocess.Popen([chrome, target])
        else:
            webbrowser.open(target)
    elif os.name == "nt":
        os.startfile(target)
    else:
        raise RuntimeError("open of apps is only supported on Windows")


def run_action(a):
    kind = a.get("do")
    if kind == "open":
        do_open(str(a["target"]))
        return f"opened {a['target']}"
    if kind == "type":
        pyautogui.write(str(a["text"]), interval=0.01)
        return f"typed {len(str(a['text']))} chars"
    if kind == "key":
        keys = [k.strip().lower() for k in str(a["keys"]).split("+") if k.strip()]
        pyautogui.hotkey(*keys)
        return f"pressed {a['keys']}"
    if kind == "click":
        pyautogui.click(int(a["x"]), int(a["y"]), button=a.get("button", "left"), clicks=int(a.get("clicks", 1)))
        return f"clicked {a['x']},{a['y']}"
    if kind == "scroll":
        pyautogui.scroll(int(a["amount"]))
        return f"scrolled {a['amount']}"
    if kind == "wait":
        s = min(float(a.get("s", 1)), 10)
        time.sleep(s)
        return f"waited {s}s"
    if kind == "screenshot":
        return "screenshot"
    raise ValueError(f"unknown action {kind!r}")


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Crane-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        origin = self.headers.get("Origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return False
        return secrets.compare_digest(self.headers.get("X-Crane-Token", ""), TOKEN)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"ok": False, "error": "bad token or origin"})
        w, h = pyautogui.size()
        self._send(200, {"ok": True, "screen": f"{w}x{h}"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": "bad token or origin"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            actions = json.loads(self.rfile.read(min(n, 1_000_000)))["actions"]
            if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
                raise ValueError("actions must be a list of at most %d items" % MAX_ACTIONS)
        except Exception as e:
            return self._send(400, {"error": f"bad request: {e}"})
        results, shot = [], None
        for a in actions:
            try:
                msg = run_action(a)
                results.append({"ok": True, "msg": msg})
                if a.get("do") == "screenshot":
                    shot = screenshot_b64()
            except pyautogui.FailSafeException:
                results.append({"ok": False, "msg": "aborted: mouse moved to a screen corner"})
                break
            except Exception as e:
                results.append({"ok": False, "msg": f"{a.get('do')}: {e}"})
                break
        self._send(200, {"results": results, "screenshot": shot})

    def log_message(self, fmt, *args):
        sys.stderr.write("[crane-hands] " + fmt % args + "\n")


if __name__ == "__main__":
    if "--print-token" in sys.argv:
        print(TOKEN)
        sys.exit(0)
    print(f"[crane-hands] listening on http://127.0.0.1:{PORT}  token file: {TOKEN_FILE}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
