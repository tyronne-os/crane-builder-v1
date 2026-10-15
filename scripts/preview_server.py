#!/usr/bin/env python3
"""
Dyad-style live preview backend for Crane Builder.
Serves an iframeable preview page and accepts live edits via POST.
Run on the VM:  python3 scripts/preview_server.py
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Crane Live Preview</title>
<style id="live-css"></style></head><body><div id="live-html"></div>
<script id="live-js"></script>
</body></html>"""

state = {"css":"", "js":"", "html":""}

class Handler(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204); self.cors(); self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/preview":
            body = BASE_HTML.replace('<style id="live-css"></style>', f'<style id="live-css">{state["css"]}</style>')
            body = body.replace('<div id="live-html"></div>', f'<div id="live-html">{state["html"]}</div>')
            body = body.replace('<script id="live-js"></script>', f'<script id="live-js">{state["js"]}</script>')
            self.send_response(200); self.cors()
            self.send_header("Content-Type","text/html")
            self.end_headers(); self.wfile.write(body.encode()); return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/api/update":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length",0))
        data = json.loads(self.rfile.read(length).decode())
        state["css"] = data.get("css","")
        state["js"]  = data.get("js","")
        state["html"]= data.get("html","")
        self.send_response(200); self.cors()
        self.send_header("Content-Type","application/json")
        self.end_headers(); self.wfile.write(b'{"ok":true}')

    def log_message(self,*args): pass

if __name__ == "__main__":
    import os
    port = int(os.environ.get("CRANE_PREVIEW_PORT","11438"))
    ThreadingHTTPServer(("127.0.0.1",port), Handler).serve_forever()
