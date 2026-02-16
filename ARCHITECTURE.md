# AURALIS — AI Architecture & Development Guide
# This document is designed for AI agents to understand and continue development.

## Identity
- **Name:** AURALIS
- **Tagline:** "Hear deeper. Create beyond."
- **Type:** AI Music Production Engine
- **Language:** Python 3.12 + TypeScript (Next.js 15)
- **Deployment:** 100% cloud — AWS EC2 `g5.xlarge` (NVIDIA A10G GPU)
- **Access:** Web browser at `http://<ec2-ip>:3000` (UI) + `:8000` (API)

## Mission
AURALIS deconstructs any professional track into its atoms, understands every
element with AI, and reconstructs it from scratch. It also creates brand new
tracks from natural language descriptions via LLM orchestration.

## Architecture — 7 Layers

```
┌─────────────────────────────────────────────┐
│                  WEB UI                      │
│  Next.js 15 · shadcn/ui · wavesurfer.js     │
│  Pages: Dashboard, Deconstructor, Creator,  │
│         Studio, Mixer, Master Suite, QC, AI │
├─────────────────────────────────────────────┤
│                 API LAYER                    │
│  FastAPI · WebSocket · Pydantic models      │
├─────────────────────────────────────────────┤
│                                             │
│  EAR          HANDS        CONSOLE          │
│  Analysis     Synthesis    Mix & Master     │
│  ─────────    ──────────   ──────────       │
│  · Demucs     · DawDreamer · Pedalboard     │
│  · RoFormer   · TorchFX    · matchering     │
│  · basic-pitch· DDSP       · convergence    │
│  · librosa    · Stable     · NVIDIA FX      │
│  · essentia     Audio                       │
│                                             │
│  GRID          BRAIN        QC              │
│  Composition   AI Engine    Quality         │
│  ──────────    ──────────   ──────────      │
│  · mido        · OpenAI     · fingerprint   │
│  · musicpy       GPT        · comparator    │
│  · Magenta     · Decision   · convergence   │
│  · YuE           Engine     · 12-dim score  │
│                                             │
└─────────────────────────────────────────────┘
```

## Directory Structure

```
~/code/auralis/
├── auralis/                    # Python package
│   ├── __init__.py             # Version: 0.1.0
│   ├── config.py               # Pydantic settings (env vars)
│   ├── ear/                    # 👂 Analysis & Deconstruction
│   │   ├── separator.py        # Demucs source separation (GPU)
│   │   ├── midi_extractor.py   # basic-pitch MIDI transcription
│   │   ├── spectral.py         # librosa deep analysis (10 bands, MFCC, key)
│   │   └── profiler.py         # Track DNA map (loudness, sections, dynamics)
│   ├── hands/                  # 🎹 Synthesis & Sound Design
│   │   ├── vst_host.py         # DawDreamer VST2/VST3 host
│   │   ├── torchfx_dsp.py      # GPU-accelerated DSP (DAFx25)
│   │   ├── ddsp_synth.py       # Google DDSP differentiable synth
│   │   ├── stable_audio.py     # Stability AI text-to-audio
│   │   └── faust_dsp.py        # Custom Faust DSP modules
│   ├── console/                # 🎚️ Mixing & Mastering
│   │   ├── fx.py               # Pedalboard FX chains + VST hosting
│   │   ├── mixer.py            # Buses, sends, EQ, compression
│   │   ├── mastering.py        # Reference-matched convergence loop
│   │   └── dsp/                # Custom: Moog filter, bitcrush, sidechain
│   ├── grid/                   # 📐 Composition & Arrangement
│   │   ├── midi.py             # MIDI read/write/generate
│   │   ├── theory.py           # Music theory (musicpy)
│   │   ├── arrangement.py      # Section-based track building
│   │   └── yue_gen.py          # YuE full-song generation
│   ├── brain/                  # 🧠 AI Intelligence
│   │   ├── agent.py            # OpenAI GPT orchestrator
│   │   └── production_ai.py    # Decision engine for production
│   ├── qc/                     # 🔍 Quality Assurance
│   │   ├── fingerprint.py      # Per-band spectral fingerprint
│   │   ├── comparator.py       # A/B track comparison
│   │   ├── convergence.py      # Mastering convergence loop
│   │   └── musical_review.py   # 12-dimension scoring
│   └── api/                    # ⚡ FastAPI Backend
│       ├── server.py           # Main app + route registration
│       ├── websocket.py        # Real-time progress (ConnectionManager)
│       └── routes/
│           └── ear.py          # POST /upload, /analyze, GET /status
├── web/                        # 🌐 Next.js 15 Frontend (TODO)
├── tests/
│   └── test_api.py             # Health check + config tests
├── scripts/
│   └── quality-check.sh        # SEC + LINT + FORMAT + TYPES + TESTS
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── pyproject.toml              # UV project config
├── .env.example                # Environment variable template
├── .gitleaks.toml              # Secret scanning config
└── README.md
```

## API Reference

### REST Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check → `{"status": "ok"}` |
| GET | `/api/info` | System capabilities and layer descriptions |
| POST | `/api/ear/upload` | Upload audio file (WAV/MP3/FLAC/AIFF) |
| POST | `/api/ear/analyze/{project_id}` | Start analysis pipeline |
| GET | `/api/ear/status/{job_id}` | Poll job progress |
| GET | `/api/ear/models` | List separation models |

### WebSocket
| Path | Description |
|------|-------------|
| `WS /ws/{project_id}` | Real-time progress: `{type, step, total, percentage, message}` |

## Data Models

### SpectralProfile (ear/spectral.py)
10-band frequency analysis, MFCC, chroma, key/scale estimation, tempo,
beat tracking, harmonic ratio, RMS energy. Used for A/B comparison.

### TrackDNA (ear/profiler.py)
Complete track identity: key, scale, tempo, EBU R128 loudness (LUFS),
true peak, loudness range, crest factor, dynamic range, sections with
energy levels, and full spectral profile. Serializable to JSON.

### SeparationResult (ear/separator.py)
Paths to separated stems (vocals, drums, bass, other), model metadata,
sample rate, duration. Supports Demucs HTDemucs, HTDemucs-FT, MDX Extra.

### MIDIExtractionResult (ear/midi_extractor.py)
MIDI file path, note count, pitch range, duration, confidence score.
Batch processing of all tonal stems.

## Technology Stack

### Currently Installed (core)
- `librosa` — Audio analysis, spectral features, beat tracking
- `pyloudnorm` — EBU R128 loudness measurement
- `soundfile` — Audio I/O (WAV, FLAC, OGG)
- `numpy`, `scipy` — Numeric computing
- `mido` — MIDI read/write
- `openai` — LLM integration
- `fastapi` + `uvicorn` — HTTP/WebSocket API
- `pydantic` + `pydantic-settings` — Data validation + env config
- `structlog` — Structured logging
- `ruff` — Linting + formatting
- `mypy` — Strict type checking
- `pytest` — Testing + coverage

### Optional (installed on EC2 with GPU)
- `[ml]`: `torch`, `demucs` — Source separation, ML inference
- `[audio]`: `pedalboard`, `matchering` — FX processing, mastering

### Planned (not yet installed)
- `essentia` — 600+ audio analysis algorithms
- `dawdreamer` — VST2/VST3 host in Python
- `basic-pitch` — Audio-to-MIDI transcription
- `musicpy` — Music theory engine
- `magenta` — AI music generation
- `TorchFX` — GPU-accelerated DSP
- `DDSP` — Differentiable synthesis
- `Stable Audio Open` — Text-to-audio
- `YuE` — Full-song generation

## Quality Standards (MeetMind-proven)

| Gate | Tool | Rule |
|------|------|------|
| SEC-001 | `gitleaks` | 0 secrets in code |
| LINT | `ruff check` | 0 errors |
| FORMAT | `ruff format` | 100% formatted |
| TYPES | `mypy --strict` | 0 type errors |
| TESTS | `pytest --cov` | ≥80% coverage |
| CODE-001 | line count | ≤500 soft / ≤800 hard per file |

### Commands
```bash
# Run all quality gates
bash scripts/quality-check.sh

# Individual checks
uv run ruff check auralis/ tests/
uv run ruff format --check auralis/ tests/
uv run mypy auralis/
uv run pytest --cov=auralis --tb=short -q

# Start API server
uv run uvicorn auralis.api.server:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

```env
AURALIS_OPENAI_API_KEY=sk-...   # OpenAI API key
AURALIS_HOST=0.0.0.0            # Server bind host
AURALIS_PORT=8000               # Server bind port
AURALIS_ENV=development         # Environment
AURALIS_PROJECTS_DIR=./projects # Where projects are stored
AURALIS_SAMPLES_DIR=./samples   # Sample library
AWS_PROFILE=mibaggy-co          # AWS profile for EC2
AWS_REGION=us-east-1            # AWS region
```

## Mastering Convergence Loop (100% Match)

```
Our Mix → matchering (reference EQ + RMS + width)
       → Render Master
       → Spectral Fingerprint vs Original
       → ≤1% deviation per band? → NO → Corrective EQ → Re-render
                                 → YES → Phase ≥0.95? → NO → Phase correction
                                                       → YES → LUFS ±0.1? → APPROVED
```

### Validation Thresholds
| Dimension | Metric | Target |
|-----------|--------|--------|
| Spectral | Per-band energy (10 bands) | ≤1% deviation |
| Dynamic | LUFS, crest factor, peak | ±0.1 LUFS |
| Stereo | Width, correlation | ±0.02 |
| Temporal | Beat alignment | ≤5ms |
| Perceptual | MFCC cosine distance | ≤0.05 |

## Development Workflow

### Adding a New Module
1. Create file in the appropriate layer directory
2. Add docstring explaining purpose
3. Add type hints (mypy --strict must pass)
4. Write tests in `tests/`
5. Run `bash scripts/quality-check.sh`
6. File must be ≤500 lines (800 hard limit)

### Adding a New API Route
1. Create route file in `auralis/api/routes/`
2. Use FastAPI `APIRouter` with prefix and tags
3. Register in `auralis/api/server.py` with `app.include_router()`
4. Add Pydantic models for request/response
5. Use `asyncio.create_task()` for long-running operations
6. Send progress via `WebSocket ConnectionManager`

### Adding a New UI Page
1. Create page in `web/app/<page-name>/page.tsx`
2. Use shadcn/ui components
3. Connect to API endpoints via fetch/axios
4. Connect to WebSocket for real-time updates
5. Follow dark theme design system

## Project Roadmap

### Phase 1 ✅ — Scaffolding + EAR [CURRENT]
- Project structure, quality gates, CI pipeline
- EAR layer: spectral analysis, profiling, separation, MIDI extraction
- API server with routes and WebSocket
- Deploy to EC2

### Phase 2 — HANDS + CONSOLE
- Synthesis engine (DawDreamer + Surge XT + TorchFX + DDSP)
- FX engine (Pedalboard + custom DSP)
- Mixing engine (buses, sends, EQ, compression)
- Mastering convergence loop

### Phase 3 — GRID + BRAIN + Creator
- MIDI/composition engine
- LLM orchestrator (OpenAI GPT)
- Creator page: describe track → LLM produces everything

### Phase 4 — Million Pieces Reconstruction
- Full deconstruction + recreation

### Phase 5 — Mono Aullador Reconstruction
- Rebuild production pipeline inside AURALIS

## Key Design Decisions
1. **Graceful degradation**: Heavy ML deps are optional — core runs without GPU
2. **Async everything**: FastAPI + asyncio for non-blocking audio processing
3. **WebSocket progress**: Real-time updates for long-running operations
4. **JSON metadata**: Every operation saves metadata for reproducibility
5. **10-band spectral comparison**: Foundation for convergence mastering
6. **EBU R128 loudness**: Industry-standard loudness measurement
7. **Section detection**: Automatic arrangement mapping for reconstruction
