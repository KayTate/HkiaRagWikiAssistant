# HKIA RAG — Hello Kitty Island Adventure Q&A Assistant

A RAG (Retrieval-Augmented Generation) application that answers natural language
questions about Hello Kitty Island Adventure using the official HKIA wiki as its
knowledge base. Features an agentic LangGraph retrieval pipeline for multi-hop
prerequisite chain resolution, ChromaDB for vector storage, MLflow for experiment
tracking, and a Gradio chat interface.

## Prerequisites

- **Python 3.12** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management
- **[Ollama](https://ollama.com/)** for local embedding and LLM inference

## Setup

### macOS

```bash
# Install uv (once)
brew install uv

# Clone and set up
git clone <repo-url>
cd hkia-rag
uv venv
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Install Ollama models (for embeddings)
ollama pull nomic-embed-text
```

### Windows

```powershell
# Install uv (once)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and set up
git clone <repo-url>
cd hkia-rag
uv venv
uv sync

# Activate the virtual environment
.venv\Scripts\activate

# Install Ollama (via winget or https://ollama.com/download)
winget install Ollama.Ollama

# Install Ollama models (for embeddings)
ollama pull nomic-embed-text
```

## Configuration

Copy the example environment file and fill in any API keys:

```bash
cp .env.example .env
```

`.env.example` documents every setting with a one-line comment;
`config/settings.py` has the canonical `Field(description=...)` for each
one. The most commonly adjusted variables:

| Variable | Default | Notes |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `openai` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | See "Embedding model drift" below |
| `LLM_PROVIDER` | `openai` | `ollama`, `openai`, or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name passed to the provider |
| `OPENAI_API_KEY` | (empty) | Required for OpenAI; also for eval LLM judges |
| `ANTHROPIC_API_KEY` | (empty) | Required for Anthropic |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `180` | Raise for 70B+ local models |
| `AGENT_MAX_ITERATIONS` | `10` | Hard ceiling on agent retrieve→extract loop |

### Embedding model drift

The agent and the ChromaDB collection must use the same embedding
model — mixing vectors from different models silently corrupts search
results. The startup sync check enforces this and **will not
auto-repair**.

If you change `EMBEDDING_MODEL` or `EMBEDDING_MODEL_VERSION`, you must
also create a new collection and re-ingest:

1. Bump `CHROMA_COLLECTION_NAME` using the convention
   `hkia_{embedding_model}_{chunking_strategy}_v{n}` (e.g.
   `hkia_nomic-embed-text_recursive_v2` →
   `hkia_nomic-embed-text_recursive_v3`).
2. Run `python sync.py --mode full` to build the new collection.
3. Point your `.env` at the new collection name.

Attempting to ingest into a collection whose stored chunks disagree
with the current configuration raises `CollectionConfigMismatchError`
with the same remediation steps in the message. The check covers
`EMBEDDING_MODEL`, `CHUNKING_STRATEGY`, `CHUNK_SIZE`, and
`CHUNK_OVERLAP` — change any of them and bump the collection name.

## Usage

### Data Ingestion

Make sure Ollama is running (`ollama serve`), then:

```bash
# Full ingestion of all wiki pages
python sync.py --mode full

# Incremental sync (only new/changed pages)
python sync.py --mode incremental

# Check ingestion progress
python sync.py --mode status
```

### Snapshot capture and replay ingestion

For reproducible chunking/embedding experiments, capture the wiki to a
Parquet file once and ingest from that file under different settings.
Every variant runs against the same source bytes, so quality differences
are attributable to config — not to wiki drift between runs.

Capture a snapshot:

```bash
# Full snapshot (every page)
python scripts/snapshot_wiki.py --output snapshots/2026-05-12.parquet

# Smoke-test cap (first N pages only)
python scripts/snapshot_wiki.py \
    --output snapshots/smoke.parquet \
    --limit 50
```

Replay the snapshot into ingestion:

```bash
python sync.py --mode replay --snapshot snapshots/2026-05-12.parquet
```

Each variant should run with its own `STATE_DB_PATH` and its own
`CHROMA_COLLECTION_NAME` so the variants coexist for side-by-side
comparison:

```bash
STATE_DB_PATH=./data/state_section_v1.db \
CHROMA_COLLECTION_NAME=hkia_nomic-embed-text_section_v1 \
CHUNKING_STRATEGY=section \
python sync.py --mode replay --snapshot snapshots/2026-05-12.parquet
```

To build every collection required for the planned E1–E4 experiments
(baseline, section-chunker variant, OpenAI-embedding variant) in one
sweep, use the wrapper script:

```bash
python scripts/run_ablation_ingestion.py
python scripts/run_ablation_ingestion.py --only e1_section   # one variant
python scripts/run_ablation_ingestion.py --force             # re-ingest existing
```

The script preflights snapshot existence and `OPENAI_API_KEY`, skips
any variant whose target collection already has chunks, and stops on
the first failure. Use the manual env-var pattern above for ad-hoc
variants outside the planned matrix.

The startup drift check verifies each collection's stored
chunking/embedding settings against the active config and refuses to
mix chunks built with different parameters (see "Embedding model drift"
above).

### Chat Interface

```bash
python app/gradio_app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

### MLflow UI

```bash
.venv/bin/mlflow ui            # macOS / Linux
.venv\Scripts\mlflow ui        # Windows
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Evaluation

Run an experiment against the golden set and log results to MLflow.
`params` are auto-populated from your active settings so each run
captures a full config snapshot.

```bash
# Baseline run with default dataset (data/eval/golden_set.json)
python scripts/run_eval.py --experiment hkia_baseline --run baseline

# Custom dataset path
python scripts/run_eval.py \
    --experiment hkia_baseline \
    --run v2 \
    --dataset data/eval/golden_set.json
```

Prerequisites:

- Ingestion has completed at least once (`python sync.py --mode status`) —
  the agent retrieves from ChromaDB, so an empty collection produces
  meaningless scores.
- `OPENAI_API_KEY` is available — the LLM judges use
  `openai:/gpt-4o-mini` regardless of which provider the agent itself
  is configured for. Either set it in your shell or in `.env`; the
  runner copies the `.env` value into the process environment so
  MLflow's OpenAI SDK calls authenticate correctly.

To run the full planned E1–E4 sweep (10 runs across 4 MLflow
experiments: chunking, embedding, LLM, retrieval top_k), use the
wrapper:

```bash
python scripts/run_experiments.py                 # all 12 runs
python scripts/run_experiments.py --dry-run       # preview matrix
python scripts/run_experiments.py --only e2_openai_embed
python scripts/run_experiments.py --start-from e4_top_k_3
```

The wrapper preflights `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (only if
needed), and the existence of every collection it will read. The sweep
stops on the first failure and prints a `--start-from` hint to resume.
Assumes `scripts/run_ablation_ingestion.py` has built the variant
collections.

Inspect runs in the MLflow UI (see above).

## Architecture

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the living
description of the system: how the ingestion / vector store / agent /
eval subsystems wire together, where to look for what, and which
invariants are load-bearing. Read it before making structural changes
or onboarding a new contributor.

[`docs/HKIA_RAG_PRD.md`](docs/HKIA_RAG_PRD.md) and
[`docs/HKIA_RAG_TDD.md`](docs/HKIA_RAG_TDD.md) capture the original
product requirements and v1 technical design respectively.

## Project Structure

```text
hkia-rag/
├── config/settings.py          # Central config via pydantic-settings
├── common/                     # Cross-package utilities
│   └── http.py                 # Shared retry predicates for HTTP clients
├── ingestion/                  # Wiki ingestion pipeline
│   ├── api_client.py           # MediaWiki API wrapper (batch)
│   ├── parser.py               # Wikitext parsing with template expansion
│   ├── chunker.py              # Recursive and section-based chunking
│   ├── embedder.py             # Ollama and OpenAI embedding providers
│   ├── state_db.py             # SQLite ingestion state tracking (bulk helpers)
│   ├── snapshot.py             # Parquet snapshot read/write for replay ingestion
│   └── pipeline.py             # Full, incremental, and replay ingestion orchestration
├── vectorstore/                # ChromaDB vector store
│   ├── client.py               # Search, upsert, and embedding-model guard
│   └── schema.py               # ChunkMetadata Pydantic model (with constraints)
├── agent/                      # LangGraph agentic retrieval
│   ├── graph.py                # State graph definition and compilation
│   ├── state.py                # AgentState dataclass
│   ├── nodes.py                # Graph node functions (route, retrieve, extract, synthesize)
│   ├── llm.py                  # LLM provider clients (Ollama, OpenAI, Anthropic) + retries
│   ├── extraction.py           # Pure text helpers: entity extraction, fence stripping
│   ├── retrieval.py            # Entity → chunks: title variants, opensearch, redirect, semantic search
│   └── prompts.py              # System prompt strings used by the LLM-driven nodes
├── eval/                       # Evaluation pipeline
│   ├── dataset.py              # Dataset loading and validation
│   ├── generate.py             # Synthetic Q&A generation
│   └── runner.py               # MLflow experiment runner with LLM judge scorers
├── app/gradio_app.py           # Gradio chat frontend
├── sync.py                     # CLI entry point for ingestion (full/incremental/status/replay)
├── scripts/run_eval.py         # CLI entry point for evaluation runs
├── scripts/snapshot_wiki.py    # CLI entry point for capturing wiki snapshots
└── tests/                      # pytest test suite (160 tests)
```

## Running Tests

```bash
pytest tests/ -v
```

## Scheduled Sync (optional)

Add to crontab for weekly incremental sync:

```bash
crontab -e
```

```bash
# Run incremental HKIA wiki sync every Sunday at 3am
0 3 * * 0 cd /path/to/hkia-rag && .venv/bin/python sync.py --mode incremental >> logs/sync.log 2>&1
```
