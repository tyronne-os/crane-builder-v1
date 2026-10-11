#!/usr/bin/env python3
"""Lets the hosted Crane Builder page talk to a local Ollama.

Listens on 127.0.0.1:11435 and forwards to 127.0.0.1:11434 (your tunnel to the GPU VM),
adding the CORS / private-network headers browsers require. Only the Crane Builder
origins below are allowed, so other websites cannot use your GPU through it.
"""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ('127.0.0.1', 11434)
LISTEN = ('127.0.0.1', 11435)
ALLOWED = ('https://aibruh-crane-builder-v1.static.hf.space', 'https://aibruh-crane-builder-v1.hf.space', 'https://huggingface.co')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def cors(self):
        origin = self.headers.get('Origin', '')
        if origin in ALLOWED:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Private-Network', 'true')

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def forward(self):
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(*UPSTREAM, timeout=600)
        try:
            conn.request(self.command, self.path, body, {'Content-Type': self.headers.get('Content-Type', 'application/json')})
            resp = conn.getresponse()
        except Exception as e:
            msg = ('Cannot reach Ollama at %s:%d (%s). Is the tunnel running?' % (UPSTREAM + (e,))).encode()
            self.send_response(502)
            self.cors()
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(resp.status)
        self.cors()
        self.send_header('Content-Type', resp.getheader('Content-Type', 'application/json'))
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        while True:
            chunk = resp.read1(8192)
            if not chunk:
                break
            self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b'0\r\n\r\n')
        conn.close()

    do_GET = do_POST = forward

    def log_message(self, fmt, *args):
        pass


if __name__ == '__main__':
    print('Crane Builder proxy: http://%s:%d -> http://%s:%d  (Ctrl+C to stop)' % (LISTEN + UPSTREAM), flush=True)
    ThreadingHTTPServer(LISTEN, Handler).serve_forever()
