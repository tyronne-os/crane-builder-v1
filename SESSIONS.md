# Session Log

## 2026-10-27 · 10:45 PM CST · New Orleans
- Space access fixed: private → public
- Local stack revived (Amanda, proxy, voice, memory, terminal, builder)
- GPU tunnel re-established — 6 models answering via L4 (incl. beryl/qwen3.8-27b-coder390 Q2/Q3/Q4)
- Model menu re-engineered:
  - HF 136-model wall → collapsed behind "HF cloud models · expand"
  - OpenCode list → collapsed behind expand row
  - IN GPU section: instance + zone + L4 badge, model count, CONNECTED dot, total GB
  - Hover tooltip: per-model size + download date (Ollama /api/tags)
- Crash fix: null-guard opencodeModels.length (TypeError on menu open)
- Deployed: f287953 on HF Space, mirrored to GitHub
