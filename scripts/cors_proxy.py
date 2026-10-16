#!/usr/bin/env python3
"""Lets the hosted Crane Builder page talk to the local model servers.

Listens on 127.0.0.1:11435 and forwards to 127.0.0.1:11434 (llama.cpp on the GPU VM),
adding the CORS / private-network headers browsers require. Only the Crane Builder
origins below are allowed, so other websites cannot use your GPU through it.

Each upstream is forwarded on its own port, so the composer can reach the CPU-only
server as well as the GPU one. The browser always talks to the proxy ports; the
raw tunnel ports stay free for other tools.

    11435 -> VM:11434   GPU  (CUDA llama-server)     <- the browser's gpuBase
    11438 -> VM:11437   CPU  (CPU-only llama-server) <- the browser's cpuBase
    11439 ->      11440  Amanda Mini (heretic gemma-1B llama-server)

Note 11438, not 11437: the tunnel already forwards 11437, so listening there
would collide with it and self-loop.
"""
import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAMS = {"11435": ("127.0.0.1", 11434), "11438": ("127.0.0.1", 11437), "11439": ("127.0.0.1", 11440)}
LISTEN = ("127.0.0.1", 11435)
LISTEN_CPU = ("127.0.0.1", 11438)
LISTEN_MINI = ("127.0.0.1", 11439)
ALLOWED = ('https://aibruh-crane-builder-v1.static.hf.space', 'https://aibruh-crane-builder-v1.hf.space', 'https://huggingface.co',
           # Loopback origins: running the builder straight off this PC
           # (http://localhost:<port>/app.html or a file:// page) is a normal way to
           # develop it. These bind to 127.0.0.1 only and cannot be reached from the
           # internet, so allowing them does not widen the proxy's exposure.
           'http://localhost:8080', 'http://127.0.0.1:8080', 'null',
           'http://localhost:5173', 'http://localhost:3000', 'http://localhost:8000')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def cors(self):
        origin = self.headers.get('Origin', '')
        # Local pages (the builder served from your own machine) are always ours.
        if origin in ALLOWED or origin.startswith('http://localhost') or origin.startswith('http://127.0.0.1'):
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
        listen_port = self.server.server_address[1]
        upstream = UPSTREAMS.get(str(listen_port))
        if upstream is None:
            # Registry port passes straight through to the registry.
            upstream = ('127.0.0.1', listen_port)
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(*upstream, timeout=600)
        try:
            conn.request(self.command, self.path, body, {'Content-Type': self.headers.get('Content-Type', 'application/json')})
            resp = conn.getresponse()
        except Exception as e:
            msg = ('Cannot reach %s:%d (%s). Is the tunnel running?' % (upstream + (e,))).encode()
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
    servers = []
    for listen in (LISTEN, LISTEN_CPU, LISTEN_MINI):
        s = ThreadingHTTPServer(listen, Handler)
        servers.append(s)
        print('Crane Builder proxy: http://%s:%d -> http://%s:%d'
              % (listen + UPSTREAMS[str(listen[1])]), flush=True)
    # First server owns the main loop; the rest serve on their own threads.
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print('  registry: http://127.0.0.1:11436/v1/models (forwarded by the tunnel)', flush=True)
    servers[0].serve_forever()
