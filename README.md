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

# Install Ollama models
ollama pull nomic-embed-text
ollama pull llama3
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

# Install Ollama models
ollama pull nomic-embed-text
ollama pull llama3
```

## Configuration

Copy the example environment file and fill in any API keys:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `openai` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `anthropic` |
| `LLM_MODEL` | `llama3` | LLM model name |
| `OPENAI_API_KEY` | (empty) | Required if using OpenAI |
| `ANTHROPIC_API_KEY` | (empty) | Required if using Anthropic |

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

### Chat Interface

```bash
python app/gradio_app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

### MLflow UI

```bash
.venv/bin/mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Project Structure

```text
hkia-rag/
├── config/settings.py          # Central config via pydantic-settings
├── ingestion/                  # Wiki ingestion pipeline
│   ├── api_client.py           # MediaWiki API wrapper (batch)
│   ├── parser.py               # Wikitext parsing with template expansion
│   ├── chunker.py              # Recursive and section-based chunking
│   ├── embedder.py             # Ollama and OpenAI embedding providers
│   ├── state_db.py             # SQLite ingestion state tracking
│   └── pipeline.py             # Full and incremental ingestion orchestration
├── vectorstore/                # ChromaDB vector store
│   ├── client.py               # Search, upsert, and embedding model guard
│   └── schema.py               # ChunkMetadata Pydantic model
├── agent/                      # LangGraph agentic retrieval
│   ├── graph.py                # State graph definition and compilation
│   ├── state.py                # AgentState dataclass
│   ├── nodes.py                # Graph node functions (route, retrieve, extract, synthesize)
│   └── tools.py                # Semantic search and page lookup tools
├── eval/                       # Evaluation pipeline
│   ├── dataset.py              # Dataset loading and validation
│   ├── generate.py             # Synthetic Q&A generation
│   ├── scorers.py              # Exact match and token overlap scorers
│   └── runner.py               # MLflow experiment runner
├── app/gradio_app.py           # Gradio chat frontend
├── sync.py                     # CLI entry point for ingestion
└── tests/                      # pytest test suite
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
