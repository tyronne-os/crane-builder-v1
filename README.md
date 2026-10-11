# Crane Builder V1

AI-powered website builder. Describe your project — Beryl builds it live.

## What it is

Crane Builder is an IDE-style web app where users describe a website in natural language, and Beryl (a voice + chat AI agent) builds it in real time. The built site renders in the center preview pane as it's being generated.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React 18 (CDN UMD) + Babel standalone — single static `index.html`, no build step |
| Voice Agent UI | Mic / speaker / end-call controls, canvas waveform (flat line until a voice model is connected), empty transcript |
| Inference Backend | `berylize-node` — GCP g2-standard-4, NVIDIA L4, us-east1-c |
| Model | berylize-7b (PyTorch 2.9 + CUDA 12.9, Ubuntu 22.04) |

## Project Structure

```
crane-builder-v1/
├── frontend/
│   ├── index.html          # Crane Builder UI — React (chat + preview + voice agent)
│   ├── uploads/
│   │   ├── beryl.jpeg      # Beryl voice agent portrait
│   │   └── hero-bg.jpeg    # Crane IT hero image
│   └── screenshots/        # UI reference screenshots
├── scripts/
│   └── audit_models.sh     # Read-only audit of models on berylize-node
└── README.md
```

## Running locally

Open `frontend/index.html` in a browser — no build step. It loads React and Babel from a CDN, so it needs an internet connection.

## Hosting

Deployed as a static Hugging Face Space (`sdk: static`): https://huggingface.co/spaces/AIBRUH/crane-builder-v1 — the Space's `index.html` and `uploads/` mirror `frontend/`.

## Roadmap

- [ ] Live avatar pipeline (replacing static photo with real-time video agent)
- [ ] Connect berylize-node inference API to chat
- [ ] Streaming site generation into center preview
- [ ] Auth + project persistence
- [ ] Export / deploy built sites
