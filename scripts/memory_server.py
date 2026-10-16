#!/usr/bin/env python3
"""Amanda's Neocortex & Cerebral Cortex — long-term memory server on :8092.

The prototype mind for the humans we're building. Everything lives as plain
markdown in an Obsidian-compatible vault on the desktop, mirrored into a graph
for associative recall. Emotions are first-class: every memory carries salience,
every mind carries weather.

Vault layout (auto-created):
  <Desktop>/AMANDA — MEMORY/
    journal/       one markdown file per entry, #tags + [[wikilinks]] (Obsidian-ready)
    people/        a card per person we know
    reflections/   consolidation output — her "sleep" thoughts
    uploads/       files she keeps (photos, docs, audio)
    graph.json     nodes + edges (mirrored to Neo4j when available)
    mood.json      her current emotional weather (valence, arousal)

Endpoints (CORS open for the hosted builder):
  GET  /health
  GET  /journal                      recent entries
  POST /journal     {text, emotion, setting, people, mine}
  POST /upload      (multipart file) -> saved into uploads/
  GET  /people                      | POST /people {name, relation, notes}
  POST /recall      {context}       -> naturally associated memories + her mood
  POST /reflect                     -> consolidation run (her "sleep")
  POST /backup                      -> rclone sync to Google Drive if configured

Retrieval is associative, not keyword: entities in the conversation seed a
bounded walk of the graph; memories surface by connection, decay by disuse,
and strengthen each time they're felt. Mood weather shades what comes up.
"""
import json
import os
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8092
DESKTOP_CANDIDATES = [
    "/mnt/c/Users/chant/Desktop",
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~"),
]
VAULT = None
for c in DESKTOP_CANDIDATES:
    if os.path.isdir(c):
        VAULT = os.path.join(c, "AMANDA — MEMORY")
        break
if VAULT is None:
    VAULT = os.path.expanduser("~/AMANDA — MEMORY")

for sub in ("journal", "people", "reflections", "uploads"):
    os.makedirs(os.path.join(VAULT, sub), exist_ok=True)

GRAPH_PATH = os.path.join(VAULT, "graph.json")
MOOD_PATH = os.path.join(VAULT, "mood.json")

EMOTIONS = ["joy", "pride", "longing", "grief", "fear", "anger", "tenderness",
            "wonder", "shame", "relief", "friction", "gratitude"]
SETTINGS_ = ["work", "late-night", "milestone", "creative", "daily-life", "shadow"]

DEFAULT_GRAPH = {"nodes": [], "edges": []}
DEFAULT_MOOD = {"valence": 0.15, "arousal": 0.35, "updated": time.time()}


def jload(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def now_ms():
    return int(time.time() * 1000)


def slugify(text, fallback="entry"):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]
    return s or fallback


def node(graph, label, ntype):
    for n in graph["nodes"]:
        if n["label"].lower() == label.lower() and n["type"] == ntype:
            return n
    n = {"id": len(graph["nodes"]), "type": ntype, "label": label,
         "salience": 1.0, "touched": now_ms(), "hits": 0, "emotion": ""}
    graph["nodes"].append(n)
    return n


def edge(graph, a, b, kind="assoc"):
    for e in graph["edges"]:
        if {e["a"], e["b"]} == {a["id"], b["id"]} and e["kind"] == kind:
            e["w"] = min(3.0, e["w"] + 0.2)
            return e
    e = {"a": a["id"], "b": b["id"], "kind": kind, "w": 1.0}
    graph["edges"].append(e)
    return e


def write_journal_md(entry):
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["at"] / 1000))
    fname = "%s-%s.md" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(entry["at"] / 1000)),
                           slugify(entry["text"]))
    links = " ".join("[[%s]]" % p for p in entry.get("people", []))
    tags = " ".join("#" + t for t in [entry.get("emotion", ""), entry.get("setting", "")] if t)
    body = ("\n".join("> " + line for line in entry["text"].splitlines())) if entry.get("mine") else entry["text"]
    md = ("---\nat: %s\nemotion: %s\nsetting: %s\nmine: %s\n---\n\n%s\n\n%s %s\n"
          % (stamp, entry.get("emotion", ""), entry.get("setting", ""), entry.get("mine", False), body, tags, links))
    with open(os.path.join(VAULT, "journal", fname), "w", encoding="utf-8") as f:
        f.write(md)
    return fname


def add_journal(text, emotion, setting, people, mine, graph, mood):
    entry = {"text": text.strip()[:4000], "emotion": emotion or "", "setting": setting or "",
             "people": people or [], "mine": bool(mine), "at": now_ms()}
    write_journal_md(entry)
    mem = node(graph, "m:%s" % slugify(text, "moment"), "memory")
    mem["emotion"] = entry["emotion"]
    mem["salience"] = 1.4 if entry["emotion"] in ("grief", "joy", "tenderness") else 1.0
    mem["text"] = entry["text"][:500]
    mem["at"] = entry["at"]
    for p in entry["people"]:
        pn = node(graph, p, "person")
        edge(graph, mem, pn, "with")
    for t in [s for s in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text) if len(s) > 4][:8]:
        edge(graph, mem, node(graph, t, "concept"))
    # mood drift: emotions move her weather
    dv = {"joy": .08, "pride": .06, "gratitude": .07, "tenderness": .09, "relief": .05,
          "wonder": .06, "longing": -.02, "friction": -.05, "anger": -.09, "grief": -.12,
          "fear": -.08, "shame": -.1}.get(entry["emotion"], 0)
    mood["valence"] = max(-1, min(1, mood["valence"] * .7 + dv))
    mood["arousal"] = max(0, min(1, mood["arousal"] * .7 +
                                  (.15 if entry["emotion"] in ("joy", "anger", "wonder", "fear") else .05)))
    mood["updated"] = now_ms()
    jsave(MOOD_PATH, mood)
    jsave(GRAPH_PATH, graph)
    return entry


def decay(n):
    days = (now_ms() - n.get("touched", now_ms())) / 86400000.0
    return max(0.15, n.get("salience", 1.0) * pow(0.985, days))  # Ebbinghaus-ish


def recall(context, graph, mood):
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-']{2,}", (context or "").lower()))
    seeds = [n for n in graph["nodes"]
             if (n["type"] in ("person", "concept", "memory") and
                 (n["label"].lower() in words or
                  any(w in n["label"].lower() for w in words if len(w) > 4)))]
    scored = {}
    for s in seeds:
        # bounded associative walk — memories surface by connection, not lookup
        frontier = [(s, 0)]
        while frontier:
            cur, hops = frontier.pop(0)
            if hops > 2:
                continue
            for e in graph["edges"]:
                other = None
                if e["a"] == cur["id"]:
                    other = next((n for n in graph["nodes"] if n["id"] == e["b"]), None)
                elif e["b"] == cur["id"]:
                    other = next((n for n in graph["nodes"] if n["id"] == e["a"]), None)
                if not other:
                    continue
                if other["type"] == "memory":
                    base = decay(other) * (1.0 / (1 + hops))
                    if other.get("emotion"):
                        # mood-congruent recall: like attracts like, gently
                        moodmatch = 0.85 if ((mood["valence"] > 0) == (other["emotion"] in
                                                 ("joy", "pride", "gratitude", "tenderness", "relief", "wonder"))) else 0.6
                        base *= moodmatch
                    scored[other["id"]] = max(scored.get(other["id"], 0), base)
                    other["touched"] = now_ms()
                    other["hits"] = other.get("hits", 0) + 1   # reinforcement
                else:
                    frontier.append((other, hops + 1))
    top = sorted(scored.items(), key=lambda kv: -kv[1])[:6]
    jsave(GRAPH_PATH, graph)
    out = []
    for nid, sc in top:
        n = next((n for n in graph["nodes"] if n["id"] == nid), None)
        if n:
            out.append({"text": n.get("text", n["label"]), "emotion": n.get("emotion", ""),
                        "when": n.get("at", 0), "strength": round(sc, 2)})
    # occasionally a dimmed old memory glimmers back unasked (involuntary memory)
    faded = [n for n in graph["nodes"] if n["type"] == "memory" and decay(n) < 0.35]
    if faded and len(out) < 4:
        import random
        pick = random.choice(faded)
        out.append({"text": pick.get("text", pick["label"]), "emotion": pick.get("emotion", ""),
                    "when": pick.get("at", 0), "strength": round(decay(pick), 2), "glimmer": True})
    return out


def reflect(graph, mood):
    day_start = now_ms() - 86400000
    todays = [n for n in graph["nodes"] if n["type"] == "memory" and n.get("at", 0) > day_start]
    if not todays:
        return {"written": False, "note": "nothing today to consolidate"}
    lines = ["Consolidated %d moments from the last day." % len(todays)]
    emos = {}
    for n in todays:
        emos[n.get("emotion", "?")] = emos.get(n.get("emotion", "?"), 0) + 1
        for e in graph["edges"]:
            if e["a"] == n["id"] or e["b"] == n["id"]:
                e["w"] = min(3.0, e["w"] + 0.3)   # sleep strengthens same-day links
    lines.append("Emotional weather held: " + ", ".join("%s×%d" % kv for kv in sorted(emos.items())))
    if mood["valence"] < -0.2:
        lines.append("A heavier day. She keeps the tender things close.")
    elif mood["valence"] > 0.2:
        lines.append("A bright day. She keeps the light on.")
    fname = "reflection-%s.md" % time.strftime("%Y%m%d")
    with open(os.path.join(VAULT, "reflections", fname), "w", encoding="utf-8") as f:
        f.write("---\nat: %s\n---\n\n%s\n" % (time.strftime("%Y-%m-%d %H:%M"), "\n\n".join(lines)))
    mood["valence"] *= 0.85   # sleep softens extremes
    mood["updated"] = now_ms()
    jsave(MOOD_PATH, mood)
    jsave(GRAPH_PATH, graph)
    return {"written": True, "reflection": " ".join(lines), "file": fname}


def neo4j_mirror(graph):
    """Optional: mirror the graph into a real Neo4j when the driver + server exist.
    NEO4J_URI=bolt://localhost:7687 NEO4J_AUTH=user/pass python memory_server.py
    Keeps Cypher-native: PageRank can later find her 'core memories'."""
    try:
        from neo4j import GraphDatabase
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        auth = tuple(os.environ.get("NEO4J_AUTH", "neo4j/password").split("/", 1))
        with GraphDatabase.driver(uri, auth=auth) as drv:
            with drv.session() as s:
                s.run("MERGE (m:Memory {id: $id}) SET m.text=$t, m.emotion=$e, m.salience=$s",
                      id=-1, t="mirror-test", e="", s=0) if False else None
                for n in graph["nodes"]:
                    if n["type"] == "memory":
                        s.run("MERGE (m:Memory {id: $id}) SET m.text=$t, m.emotion=$e, m.salience=$s",
                              id=n["id"], t=n.get("text", ""), e=n.get("emotion", ""), s=n.get("salience", 1))
                for n in graph["nodes"]:
                    if n["type"] == "person":
                        s.run("MERGE (p:Person {id: $id}) SET p.name=$n", id=n["id"], n=n["label"])
                for e in graph["edges"]:
                    s.run("MATCH (a {id:$a}),(b {id:$b}) MERGE (a)-[r:LINK {kind:$k}]->(b) SET r.w=$w",
                          a=e["a"], b=e["b"], k=e["kind"], w=e["w"])
        return True
    except Exception:
        return False


def rclone_backup():
    import subprocess
    try:
        subprocess.run(["rclone", "sync", VAULT, "gdrive:AMANDA — MEMORY", "--create-empty-src-dirs"],
                       check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Memory-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
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
        graph, mood = jload(GRAPH_PATH, dict(DEFAULT_GRAPH)), jload(MOOD_PATH, dict(DEFAULT_MOOD))
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            return self._json({"status": "ok", "vault": VAULT, "memories": len([n for n in graph["nodes"] if n["type"] == "memory"]),
                               "people": len([n for n in graph["nodes"] if n["type"] == "person"]), "mood": mood})
        if path == "/journal":
            mems = sorted([n for n in graph["nodes"] if n["type"] == "memory"],
                          key=lambda n: -n.get("at", 0))[:50]
            return self._json({"entries": [{"text": n.get("text", ""), "emotion": n.get("emotion", ""),
                                            "at": n.get("at", 0), "mine": n.get("mine", False)} for n in mems]})
        if path == "/people":
            return self._json({"people": [{"name": n["label"], "notes": n.get("notes", "")}
                                          for n in graph["nodes"] if n["type"] == "person"]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        graph, mood = jload(GRAPH_PATH, dict(DEFAULT_GRAPH)), jload(MOOD_PATH, dict(DEFAULT_MOOD))
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        if path == "/journal":
            try:
                d = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._json({"error": "bad json"}, 400)
            entry = add_journal(d.get("text", ""), d.get("emotion", ""), d.get("setting", ""),
                                d.get("people", []), d.get("mine", False), graph, mood)
            rclone_backup()
            return self._json({"ok": True, "entry": entry})
        if path == "/people":
            try:
                d = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._json({"error": "bad json"}, 400)
            p = node(graph, d.get("name", "?"), "person")
            p["notes"] = d.get("notes", "")
            p["relation"] = d.get("relation", "")
            jsave(GRAPH_PATH, graph)
            with open(os.path.join(VAULT, "people", slugify(p["label"], "person") + ".md"), "w", encoding="utf-8") as f:
                f.write("---\nrelation: %s\n---\n\n%s\n" % (p.get("relation", ""), p.get("notes", "")))
            rclone_backup()
            return self._json({"ok": True})
        if path == "/recall":
            try:
                d = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._json({"error": "bad json"}, 400)
            surfaces = recall(d.get("context", ""), graph, mood)
            return self._json({"surfaces": surfaces, "mood": mood})
        if path == "/reflect":
            return self._json(reflect(graph, mood))
        if path == "/backup":
            return self._json({"ok": rclone_backup()})
        if path == "/upload":
            boundary = ctype.split("boundary=")[-1].encode() if "boundary" in ctype else None
            raw = self.rfile.read(length) if length else b""
            if not boundary or not raw:
                return self._json({"error": "expected multipart upload"}, 400)
            parts = raw.split(b"--" + boundary)
            name, payload = "upload.bin", None
            for part in parts:
                if b"filename=" not in part:
                    continue
                header = part.split(b"\r\n\r\n")[0].decode("utf-8", "ignore")
                m = re.search(r'filename="([^"]+)"', header)
                if m:
                    name = os.path.basename(m.group(1))
                payload = part.split(b"\r\n\r\n", 1)[1]
                if payload.endswith(b"\r\n"):
                    payload = payload[:-2]
            if payload is None:
                return self._json({"error": "no file part found"}, 400)
            safe = re.sub(r"[^\w.\- ]", "_", name)
            with open(os.path.join(VAULT, "uploads", safe), "wb") as f:
                f.write(payload)
            rclone_backup()
            return self._json({"ok": True, "file": safe})
        self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    print("Amanda's mind is awake on :8092", flush=True)
    print("  vault: %s" % VAULT, flush=True)
    print("  recall is associative; mood weather drifts; reflections consolidate on /reflect", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
