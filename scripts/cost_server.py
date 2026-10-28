#!/usr/bin/env python3
"""Crane cost server — real GCP spend tracking on :11444.
Reads status from a JSON file updated by a bash poller (gcloud is a Windows
batch file that Python can't call directly)."""
import json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_FILE = "/tmp/opencode/gcp_state.json"
RATE = 0.22

def get_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"instance": "berylize-node", "status": "UNKNOWN", "hours": 0, "cost": 0, "rate": RATE, "gpu": "L4", "zone": "us-east1-c"}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.send_header("Content-Length", "0"); self.end_headers()
    def do_GET(self):
        s = get_state()
        body = json.dumps(s).encode()
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 11444), H).serve_forever()
