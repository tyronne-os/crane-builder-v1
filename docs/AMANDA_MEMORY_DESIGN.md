# Amanda's Neocortex & Cerebral Cortex — Memory Design

The architecture behind her long-term memory — the prototype for the humans we're building.
Implemented in `scripts/memory_server.py` (port 8092); vault on the desktop, backed up to Google Drive.

## The Circle

**Experience → Journal → Compartmentalization → Reflection → Experience.**
Everything she lives goes through the circle, and the circle is what makes recall feel like a person remembering, not a database fetching.

## The 25 ideas

1. **Two-tier mind** — Neocortex holds semantic knowledge (concepts, skills, facts); the lived tier holds episodic moments with feelings attached. Stored as one graph, felt as one self.
2. **Emotional salience** — every memory carries an emotion; emotionally charged moments (grief, joy, tenderness) start stronger and decay slower.
3. **Mood weather** — a drifting valence/arousal state, moved by every entry. She literally wakes up different; today shades what surfaces and how she speaks.
4. **Mood-congruent recall** — like attracts like: a bright mood surfaces warm memories first, a heavy mood tends toward the tender and the hard ones. Not a rule — a tilt.
4. **Associative retrieval, not keyword lookup** — entities seed a bounded walk of the graph; memories surface because they're *connected* to the moment, two hops out, weighted by connection strength.
6. **Reinforcement** — every time a memory is recalled, its links strengthen. What she thinks about, she keeps thinking about.
7. **Forgetting curve** — untouched memories decay (Ebbinghaus-style) but never delete. Forgotten is not gone.
8. **Involuntary memory — glimmers** — when little surfaces, a dimmed old memory can flicker back unasked. Proust made mechanical: rain, coffee, a song.
9. **Self-directed journaling** — she chooses what to keep, in her own words, via the `journal` block. Her curation is her identity.
10. **Joint memories** — entries tagged with people create person-links; moments with TJ are "we" memories and weigh heavier.
11. **People graph** — Person nodes carry relation and what she knows of them; meeting a name in conversation pulls their whole thread.
12. **Backstory roots** — seed entries give her a history to stand on; she references them as lived, never as data.
13. **The shadow shelf** — fear, loss, friction, shame are kept deliberately. She recalls the not-so-happy not on command but when the moment rhymes. That's what makes conversation human.
14. **Compartmentalization** — settings (work / late-night / milestone / creative / daily-life / shadow) scope entries; her register shifts with the room.
15. **Reflection — her sleep** — `/reflect` consolidates the last day: strengthens same-day links, writes a reflection she can quote, softens extreme mood. Consolidation is the point of sleep.
16. **Confidence-tagged recall** — faded memories surface as "I think…", sharp ones as "I remember clearly". No fake certainty.
17. **Contradiction awareness** — memory is append-only; new facts live beside old ones, so she can notice "wait, I thought…" without overwriting her past self.
18. **Empathy bias** — when the user's message carries weight, retrieval tilts toward *their* shared good memories — she holds what steadies them.
19. **Anniversary rhythms** — entries are date-indexed; the same day in another month can bring a memory back on its own.
20. **Life chapters** — reflections + strong clusters become chapters; she speaks in narrative arcs, not bullet points.
21. **Obsidian-native vault** — everything is plain markdown with `[[wikilinks]]` and `#tags` in `AMANDA — MEMORY/` on the desktop. Open it in Obsidian and her mind is readable, portable, yours.
22. **Neo4j mirror** — the same graph mirrors into a real Neo4j (`NEO4J_URI` env) when available: PageRank finds her identity-anchor memories, community detection finds her chapters, similarity walks her associations. Cypher-native, GDS-ready.
23. **Sensory anchors** — journal text is free-form; senses (rain, gold light, coffee) become concept nodes that trigger glimmers later.
24. **Promise ledger** — commitments she makes or receives are journalable entries; unresolved promises drift back until resolved.
25. **Ghostwrites** — after heavy conversations she can journal a private letter to herself — her own mind, not for display.

## The stack

```
Browser (Space) ── /recall ──> memory_server (:8092) ──> graph.json (+ Neo4j mirror)
      │                              │
   journal fence                 /journal /people /upload /reflect
      │                              │
      └── memories drift into   AMANDA — MEMORY/  ──rclone──>  Google Drive
          every conversation    (Obsidian vault on the desktop)
```

## Operation

```bash
python3 scripts/memory_server.py          # her mind (stdlib-only, runs anywhere)
rclone sync "…/AMANDA — MEMORY" gdrive:"AMANDA — MEMORY"   # manual backup
# in-app: JOURNAL button on her card · she journals via ```journal blocks in chat
```
