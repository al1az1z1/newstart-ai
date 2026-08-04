# NewStart AI — Government Document Routing (MVP)

A small, fixed-dataset capstone comparing three ways to route a government-agency document
(USCIS, California DMV, SSA, IRS) to the correct agency: a fine-tuned BERT classifier, an
LLM classifier (Gemini), and an LLM+RAG classifier — evaluated on the same frozen test set.

Full architecture, interfaces, data flow, and phased implementation plan:
[`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) (derived from
[`../NewStart_AI_MVP.md`](../NewStart_AI_MVP.md)).

## Layout

- `notebooks/00_data_acquisition/` — existing crawlers (USCIS/DMV/IRS/SSA) that build
  `data/processed/final_dataset.csv` (754 labeled documents). Unchanged from earlier work.
- `notebooks/01`–`10` — the research pipeline: validation, EDA, one frozen split, BERT
  fine-tuning + evaluation, LLM evaluation, RAG index + evaluation, comparison, summary.
  **Implemented and executed end to end** — see Status below.
- `src/newstart_ai/` — reusable services shared by notebooks and the API (no logic
  duplicated in either place).
- `backend/` — FastAPI demo API (Research Results + Random Form Routing Demo endpoints).
  Reads stored research artifacts and calls the same `src/newstart_ai` services the
  notebooks use — no training, upload, or database. **Implemented and verified.**
- `frontend/` — React + Vite demo UI with two pages (Research Results, Random Form Routing
  Demo). **Source written; not installed or run in this environment** — see the note below.
- `configs/base.yaml`, `bert.yaml`, `llm.yaml`, `rag.yaml` — experiment configuration.

## Setup (research notebooks)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements/notebooks.txt   # for running the research notebooks
cp .env.example .env   # then set GEMINI_API_KEY
```

This project was actually developed and run against a dedicated `newstart-ai` conda
environment with GPU-accelerated PyTorch (see Status below) — `requirements/notebooks.txt`
lists what that environment needs; adjust the PyTorch install line for CPU-only if no GPU is
available.

## Running the demo app

### Backend (FastAPI)

Requires: `requirements/base.txt` installed (adds `fastapi`, `uvicorn`, `chromadb`, `torch`,
`transformers`, `google-genai`, etc. on top of the notebook requirements) and a completed
research pipeline (`artifacts/models/`, `artifacts/reports/`, `artifacts/vector_stores/` must
already exist — run notebooks 01–10 first if they don't).

```bash
cd newstart_ai_benchmark
pip install -r requirements/base.txt
cp .env.example .env   # then set GEMINI_API_KEY (same key the notebooks use)

# src/ and the project root both need to be importable:
# Linux/macOS:
PYTHONPATH="src:." python -m uvicorn backend.app.main:app --reload
# Windows PowerShell:
$env:PYTHONPATH = "src;."; python -m uvicorn backend.app.main:app --reload
```

The API serves on `http://localhost:8000`. Endpoints:

- `GET /api/health` — liveness check.
- `GET /api/research-results` — reads notebooks 09/10's saved comparison tables, metrics,
  confusion matrices, error analysis, and reproducibility manifest. No computation happens
  here.
- `POST /api/demo/random-form` — picks one random row from the frozen `test.csv`, runs BERT,
  the LLM, and LLM+RAG on that exact text, routes to a `GuidanceAgent` using the configured
  `demo.default_routing_method`, and returns everything (predictions, ground truth, routed
  agency, guidance text).

Both endpoints were run and verified end to end against the real trained BERT artifact and
real Gemini calls (see Status below) — including over real HTTP via `uvicorn`, not only
through an in-process test client.

### Frontend (React + Vite)

```bash
cd newstart_ai_benchmark/frontend
npm install
cp .env.example .env   # only needed if the backend isn't at localhost:8000
npm run dev
```

Opens on `http://localhost:5173`, calling the backend at `http://localhost:8000/api` by
default (`VITE_API_BASE_URL` in `.env` to override).

**Important:** the frontend's source code was written but this development environment has
no Node.js/npm installed, so `npm install`/`npm run dev` have **not** been run here — the
React code is untested. Before relying on it, run it locally and confirm both pages
(Research Results, Random Form Routing Demo) render against the live backend.

## Status

**Research pipeline (notebooks 00–10) is complete**, run end to end against the real
dataset in the `newstart-ai` conda environment (GPU-accelerated BERT, real Gemini API calls
for LLM/LLM+RAG/embeddings):

- Frozen split: train 482 / validation 121 / test 151 (seed 42).
- Test-set macro F1: **BERT 0.974**, LLM 0.969, LLM+RAG 0.969 (LLM+RAG made identical
  predictions to plain LLM — no measurable benefit from retrieval on this dataset).
- Long-document strategy resolved to `first_512`; demo default routing method resolved to
  `bert` — both decided by validation macro F1 comparison, written to `configs/base.yaml`.
- Two of three total test-set errors traced to upstream dataset-quality issues (one
  mislabel, one empty PDF extraction), not model weaknesses — see notebook 09.
- Full detail: `notebooks/10_research_summary.ipynb`,
  `artifacts/reports/reproducibility_manifest.json`.

**Phase 2 (demo app) status:**

- **Backend:** fully implemented and verified — both endpoints tested via `TestClient` and
  again over real HTTP (`uvicorn` + `curl`), using the real trained BERT artifact and real
  Gemini calls. Along the way, found and fixed a real bug: `gemini-3.6-flash`'s default
  "thinking" behavior was silently truncating the guidance agent's answers before any visible
  text was produced (`finish_reason: MAX_TOKENS` with ~1-6 visible tokens); fixed by setting
  `thinking_level=MINIMAL` for the guidance path (classification calls were unaffected since
  they don't cap `max_output_tokens`).
- **Frontend:** source fully written (two pages, API client, routing) but **not installed or
  run** — this environment has no Node.js/npm. Treat it as unverified until run locally.
