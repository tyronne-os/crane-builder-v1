# Crane Builder V1

AI-powered website builder. Describe your project — Beryl builds it live.

## What it is

Crane Builder is an IDE-style web app where users describe a website in natural language, and Beryl (a voice + chat AI agent) builds it in real time. The built site renders in the center preview pane as it's being generated.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vanilla HTML/CSS/JS — no framework |
| Voice Agent UI | HTML5 Canvas oscilloscope, typewriter transcript |
| Inference Backend | `berylize-node` — GCP g2-standard-4, NVIDIA L4, us-east1-c |
| Model | berylize-7b (PyTorch 2.9 + CUDA 12.9, Ubuntu 22.04) |

## Project Structure

```
crane-builder-v1/
├── frontend/
│   ├── index.html          # Crane Builder UI (chat + preview + voice agent)
│   ├── uploads/
│   │   ├── beryl.jpeg      # Beryl voice agent portrait
│   │   └── hero-bg.jpeg    # Crane IT hero image
│   └── screenshots/        # UI reference screenshots
└── README.md
```

## Running locally

Just open `frontend/index.html` in a browser — no build step, no dependencies.

## Roadmap

- [ ] Live avatar pipeline (replacing static photo with real-time video agent)
- [ ] Connect berylize-node inference API to chat
- [ ] Streaming site generation into center preview
- [ ] Auth + project persistence
- [ ] Export / deploy built sites
