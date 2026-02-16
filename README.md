# 🎛️ AURALIS — AI Music Production Engine

> *"Hear deeper. Create beyond."*

**A software that doesn't exist** — an AI engine that deconstructs any professional track into its atoms, understands every element, and reconstructs it from scratch. 100% cloud-based on EC2, controllable from any browser.

## Features

- 👂 **EAR** — Deconstruct any track (Mel-Band RoFormer, HTDemucs v4, basic-pitch)
- 🎹 **HANDS** — Synthesis engine (DawDreamer + VSTs, TorchFX GPU DSP, DDSP)
- 🎚️ **CONSOLE** — Mix & master with reference-matched convergence loop
- 📐 **GRID** — MIDI & composition (musicpy, Magenta, YuE)
- 🧠 **BRAIN** — LLM-powered production decisions (OpenAI GPT)
- 🔍 **QC** — 12-dimension quality scoring with spectral fingerprint
- 🌐 **Web UI** — Ultra-modern Next.js dashboard controlling everything

## Quick Start

```bash
# Install dependencies
uv sync --all-groups

# Copy env file and add your OpenAI API key
cp .env.example .env

# Run quality checks
bash scripts/quality-check.sh

# Start API server
uv run uvicorn auralis.api.server:app --reload

# Run tests
uv run pytest
```

## Architecture

```
auralis/
├── ear/       # 👂 Analysis & Deconstruction
├── hands/     # 🎹 Synthesis & Sound Design
├── console/   # 🎚️ Mix & Master
├── grid/      # 📐 Composition & Arrangement
├── brain/     # 🧠 AI Intelligence
├── qc/        # 🔍 Quality Assurance
└── api/       # ⚡ FastAPI + WebSocket
```

## License

MIT
