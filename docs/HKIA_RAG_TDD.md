# Technical Design Document

## Hello Kitty Island Adventure — RAG Q&A Application

**Version:** 1.0  
**Status:** Draft  
**Author:** Kaycee  
**Date:** April 2026  
**Reference:** HKIA_RAG_PRD.md

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Environment and Dependencies](#2-environment-and-dependencies)
3. [Configuration](#3-configuration)
4. [Ingestion Pipeline](#4-ingestion-pipeline)
5. [Vector Store and Collection Management](#5-vector-store-and-collection-management)
6. [LangGraph Agent](#6-langgraph-agent)
7. [Evaluation Pipeline](#7-evaluation-pipeline)
8. [Gradio Frontend](#8-gradio-frontend)
9. [Sync Scheduler](#9-sync-scheduler)
10. [Testing Strategy](#10-testing-strategy)
11. [Dependency Decisions Log](#11-dependency-decisions-log)

---

## 1. Repository Structure

```text
hkia-rag/
├── README.md
├── pyproject.toml              # uv project manifest and dependencies
├── .python-version             # pinned Python version for uv
├── .env.example                # template for required environment variables
├── .gitignore
│
├── config/
│   └── settings.py             # central config loaded from environment
│
├── ingestion/
│   ├── __init__.py
│   ├── api_client.py           # MediaWiki API wrapper
│   ├── parser.py               # wikitext → plain text via mwparserfromhell
│   ├── chunker.py              # chunking strategies
│   ├── embedder.py             # embedding model abstraction
│   ├── state_db.py             # SQLite ingestion state management
│   └── pipeline.py             # orchestrates full and incremental ingestion
│
├── vectorstore/
│   ├── __init__.py
│   ├── client.py               # ChromaDB client and collection management
│   └── schema.py               # chunk metadata schema and validation
│
├── agent/
│   ├── __init__.py
│   ├── graph.py                # LangGraph graph definition
│   ├── state.py                # agent state dataclass
│   ├── nodes.py                # individual graph node functions
│   └── tools.py                # retrieval tool definitions
│
├── eval/
│   ├── __init__.py
│   ├── dataset.py              # dataset loading and validation
│   ├── generate.py             # LLM-based synthetic Q&A generation
│   ├── scorers.py              # RAGAS-style scorer definitions
│   └── runner.py               # MLflow experiment runner
│
├── app/
│   ├── __init__.py
│   └── gradio_app.py           # Gradio chat interface
│
├── sync.py                     # CLI entry point for manual / cron sync
├── data/
│   ├── ingestion_state.db      # SQLite state file (gitignored)
│   └── eval/
│       └── golden_set.json     # hand-written eval questions (versioned)
│
├── chroma_data/                # ChromaDB persistence directory (gitignored)
├── mlruns/                     # MLflow tracking store (gitignored)
│
└── tests/
    ├── __init__.py
    ├── test_ingestion_idempotency.py
    ├── test_agent_cycle_detection.py
    └── test_embedding_version_guard.py
```

---

## 2. Environment and Dependencies

### 2.1 Package Manager

**uv** is used for environment and dependency management. uv is a drop-in replacement for pip and venv with significantly faster dependency resolution and installation. It is the recommended choice for new Python ML projects as of 2025.

**Tradeoff acknowledged:** uv is a newer tool and may be unfamiliar to some collaborators. pip + venv would be more universally understood. uv is chosen here because its workflow is nearly identical to pip, its speed is meaningfully better during iterative development, and it signals currency with the Python ecosystem to technical reviewers.

Setup:

```bash
# Install uv (once, via Homebrew)
brew install uv

# Create project and virtual environment
uv init hkia-rag
cd hkia-rag
uv venv

# Install dependencies
uv add <package>

# Activate environment
source .venv/bin/activate
```

### 2.2 Python Version

Confirm installed version with `python --version`. Record the version in `.python-version` at project root for uv to pin it. Either 3.11 or 3.12 is acceptable; 3.10 and below are not supported due to LangGraph requirements.

### 2.3 Core Dependencies

| Package | Version Constraint | Purpose |
| --- | --- | --- |
| `langraph` | `>=0.2` | Agent orchestration |
| `langchain-core` | `>=0.2` | Tool and message primitives |
| `chromadb` | `>=0.5` | Vector store |
| `mwparserfromhell` | `>=0.6` | Wikitext parsing |
| `mlflow` | `>=2.14` | Experiment tracking and tracing |
| `gradio` | `>=4.0` | Chat frontend |
| `ollama` | `>=0.2` | Local LLM and embedding client |
| `openai` | `>=1.0` | OpenAI embeddings (experiment) |
| `tenacity` | `>=8.0` | Retry logic for API calls |
| `pydantic` | `>=2.0` | Config and schema validation |
| `pytest` | `>=8.0` | Test framework (dev dependency) |
| `pytest-mock` | `>=3.0` | Mocking for ingestion tests (dev dependency) |

---

## 3. Configuration

All configuration is loaded from environment variables via a central `config/settings.py` using Pydantic's `BaseSettings`. No hardcoded values anywhere in the codebase. Sensitive values (API keys) are never committed to git.

### 3.1 settings.py

```python
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # MediaWiki
    wiki_base_url: str = "https://hellokittyislandadventure.wiki.gg"
    wiki_api_url: str = "https://hellokittyislandadventure.wiki.gg/api.php"
    wiki_request_delay_seconds: float = 0.75

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "hkia_v1"  # bump on embedding model change

    # Embedding
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_model_version: str = "v1.5"
    openai_api_key: str = ""  # required if embedding_provider = openai
    openai_embedding_batch_size: int = 100

    # LLM
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "llama3"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # LangGraph
    agent_max_iterations: int = 10

    # SQLite
    state_db_path: str = "./data/ingestion_state.db"

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "hkia_rag"

    # Chunking
    chunking_strategy: Literal["recursive", "section"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### 3.2 .env.example

```text
WIKI_BASE_URL=https://hellokittyislandadventure.wiki.gg
CHROMA_COLLECTION_NAME=hkia_v1
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_MODEL_VERSION=v1.5
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

---

## 4. Ingestion Pipeline

### 4.1 SQLite State Schema

`ingestion/state_db.py` manages a SQLite database at `settings.state_db_path`.

```sql
CREATE TABLE IF NOT EXISTS page_ingestion_state (
    page_title      TEXT PRIMARY KEY,
    revision_id     INTEGER NOT NULL,
    status          TEXT NOT NULL CHECK(status IN ('pending', 'in_progress', 'complete')),
    embedding_model TEXT NOT NULL,
    updated_at      TEXT NOT NULL   -- ISO 8601 timestamp
);
```

**Key operations:**

```python
def upsert_page(page_title, revision_id, status, embedding_model) -> None
def get_page(page_title) -> dict | None
def get_pages_by_status(status: str) -> list[dict]
def get_pages_with_stale_embedding_model(current_model: str) -> list[dict]
    # returns all pages where embedding_model != current_model
    # used by startup sync check to identify pages needing re-ingestion
def mark_complete(page_title) -> None
def mark_pending(page_title, revision_id) -> None
```

### 4.2 MediaWiki API Client

`ingestion/api_client.py` wraps all MediaWiki API calls. All requests sleep `settings.wiki_request_delay_seconds` between calls. Retries on transient HTTP errors via `tenacity`.

```python
def get_all_page_titles() -> list[str]
    # action=query&list=allpages&aplimit=500&format=json
    # paginates via apcontinue until exhausted

def get_page_wikitext(page_title: str) -> str
    # action=parse&page={title}&prop=wikitext&format=json

def get_page_revision_id(page_title: str) -> int
    # action=query&titles={title}&prop=revisions&rvprop=ids&format=json
```

### 4.3 Wikitext Parser

`ingestion/parser.py` converts raw wikitext to plain text and extracts section structure.

```python
def parse_wikitext(wikitext: str) -> str
    # strips templates, links, markup via mwparserfromhell
    # returns clean plain text

def extract_sections(wikitext: str) -> list[dict]
    # returns [{"heading": str, "content": str}, ...]
    # used by section-based chunking strategy
```

### 4.4 Chunker

`ingestion/chunker.py` implements both chunking strategies. Strategy is selected at runtime from `settings.chunking_strategy`.

```python
def chunk_text(text: str, strategy: str, chunk_size: int, overlap: int) -> list[str]
    # dispatches to recursive_chunk() or section_chunk()

def recursive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]
    # uses LangChain RecursiveCharacterTextSplitter
    # splits on ["\n\n", "\n", " ", ""] in order

def section_chunk(sections: list[dict], chunk_size: int, overlap: int) -> list[str]
    # splits each section independently
    # prepends section heading to each chunk for context
    # falls back to recursive_chunk for sections exceeding chunk_size
```

### 4.5 Embedder

`ingestion/embedder.py` abstracts over Ollama and OpenAI embedding providers.

```python
def embed_chunks(chunks: list[str]) -> list[list[float]]
    # dispatches to ollama_embed() or openai_embed()
    # returns list of embedding vectors

def ollama_embed(chunks: list[str]) -> list[list[float]]
    # calls Ollama API with settings.embedding_model
    # embeds one at a time (Ollama does not support batch)

def openai_embed(chunks: list[str]) -> list[list[float]]
    # batches into settings.openai_embedding_batch_size
    # retries on HTTP 429 with exponential backoff via tenacity
    # raises EmbeddingModelMismatchError if collection was built
    # with a different embedding model (see Section 5.2)
```

### 4.6 Ingestion Pipeline Orchestration

`ingestion/pipeline.py` is the top-level orchestrator. Both full and incremental ingestion use the same per-page logic — they differ only in which pages are in scope.

**Full ingestion:**

```text
1. Fetch all page titles from MediaWiki API
2. For each title: upsert status = pending in SQLite
3. Run per-page ingestion loop (see below)
```

**Incremental ingestion:**

```text
1. Fetch all page titles and current revision IDs from MediaWiki API
2. For each title:
   a. If not in SQLite → mark pending (new page)
   b. If revision_id changed → mark pending (updated page)
   c. If same revision_id and status = complete → skip
3. Run per-page ingestion loop for all pending pages
```

**Per-page ingestion loop:**

```text
1. Mark page status = in_progress
2. Fetch wikitext via API
3. Parse wikitext → plain text + sections
4. Chunk according to settings.chunking_strategy
5. Embed chunks
6. Delete existing chunks for this page from ChromaDB (idempotency)
7. Insert new chunks with full metadata (including embedding_model field)
8. Mark page status = complete in SQLite with embedding_model field
```

Steps 7 and 8 always execute together in the same logical operation — ChromaDB chunks and the SQLite row are always written with the same `embedding_model` value. Drift between the two stores can only occur if the process crashes between steps 7 and 8. This is repaired automatically by the startup sync check described in Section 5.2.

If any step raises an exception, status remains `in_progress`. On retry the page is reprocessed from step 1, and the delete in step 6 ensures no duplicate chunks accumulate.

---

## 5. Vector Store and Collection Management

### 5.1 Chunk Document Schema

Every document stored in ChromaDB has the following metadata:

```python
class ChunkMetadata(BaseModel):
    source_title: str           # wiki page title
    source_url: str             # full URL to wiki page
    section: str                # section heading or "" if unsectioned
    category: str               # wiki category (e.g. "Quests")
    chunk_index: int            # position within source page
    revision_id: int            # MediaWiki revision ID at ingestion
    ingested_at: str            # ISO 8601 timestamp
    embedding_model: str        # e.g. "nomic-embed-text:v1.5"
    chunking_strategy: str      # "recursive" or "section"
    chunk_size: int
    chunk_overlap: int
```

### 5.2 Collection Versioning and Embedding Model Guard

`vectorstore/client.py` and `ingestion/pipeline.py` together enforce consistency between the ChromaDB `embedding_model` chunk metadata and the SQLite `embedding_model` column. These two stores operate at different granularities — ChromaDB tracks chunks, SQLite tracks pages — and serve different purposes. ChromaDB is the source of truth for retrieval; SQLite is the source of truth for ingestion orchestration.

**Source of truth:** ChromaDB is authoritative for which embedding model produced each chunk. SQLite is authoritative for page-level ingestion status. Neither is derived from the other; they are written together during the per-page ingestion loop (Section 4.6) and validated together at startup.

**Startup sync check** — runs at the start of every ingestion run and on application startup. Repairs any drift before ingestion proceeds.

```python
def run_startup_sync_check() -> None
    # Step 1: verify ChromaDB collection embedding model
    #   samples 10 random chunks from the collection
    #   if any chunk has embedding_model != current settings:
    #     raise EmbeddingModelMismatchError with instructions:
    #       1. Update settings.chroma_collection_name to a new version (e.g. hkia_v2)
    #       2. Run full ingestion to build the new collection
    #       3. Update config to point the app at the new collection
    #   this case requires operator intervention — it means the collection
    #   was built with a different model than currently configured

    # Step 2: repair SQLite rows with stale embedding_model
    #   query SQLite for pages where embedding_model != current settings
    #   for each stale page: mark status = pending
    #   these pages will be re-ingested in the current run with the correct model
    #   this repairs drift caused by crashes between ChromaDB write and SQLite write
```

**When EmbeddingModelMismatchError is raised vs. auto-repaired:**

- **Auto-repaired (SQLite drift):** SQLite rows with a stale `embedding_model` are silently reset to `pending`. This covers the crash-between-writes scenario and requires no operator action.
- **Operator intervention required (ChromaDB mismatch):** If the ChromaDB collection itself contains chunks from a different model than current settings, the entire collection is invalid for retrieval. This cannot be auto-repaired in place because vectors from different embedding spaces cannot coexist in one collection. The operator must create a new versioned collection and run full ingestion.

**Collection naming convention:** `hkia_v{n}` where n is incremented on any embedding model change. Current baseline is `hkia_v1`. The active collection name is always set via `settings.chroma_collection_name` so cutover is a config change, not a code change.

### 5.3 ChromaDB Client Interface

```python
def get_or_create_collection(name: str) -> chromadb.Collection

def upsert_chunks(
    page_title: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[ChunkMetadata]
) -> None

def delete_chunks_by_source(page_title: str) -> None
    # deletes all chunks where metadata.source_title == page_title

def semantic_search(
    query_embedding: list[float],
    top_k: int,
    where: dict | None = None   # optional metadata filter
) -> list[dict]                  # returns chunks with text and metadata

def get_page_by_title(page_title: str) -> list[dict]
    # exact match on metadata.source_title
    # returns all chunks for that page ordered by chunk_index
```

---

## 6. LangGraph Agent

### 6.1 Agent State

`agent/state.py` defines the state object passed between all graph nodes.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentState:
    question: str                          # original user question
    messages: list[Any] = field(default_factory=list)  # conversation history
    retrieved_context: list[dict] = field(default_factory=list)  # all chunks retrieved so far
    resolved_entities: dict[str, list[dict]] = field(default_factory=dict)  # entity name → chunks
    prerequisite_chain: list[str] = field(default_factory=list)  # ordered chain built so far
    visited: set[str] = field(default_factory=set)  # cycle prevention
    iteration_count: int = 0               # tracks against agent_max_iterations
    final_answer: str = ""                 # populated by synthesis node
    needs_more_retrieval: bool = True      # controls loop continuation
```

### 6.2 Retrieval Tools

`agent/tools.py` exposes two tools to the LangGraph agent.

#### Semantic search tool

```python
@tool
def semantic_search(query: str, category_filter: str | None = None) -> list[dict]:
    """
    Search the HKIA wiki by semantic similarity. Use for broad questions
    where the exact page title is not known. Returns the most relevant
    chunks across all wiki pages. Optionally filter by category
    (e.g. 'Quests', 'Characters', 'Items').
    """
```

#### Exact page lookup tool

```python
@tool
def get_page(page_title: str) -> list[dict]:
    """
    Retrieve all content chunks for a specific wiki page by exact title.
    Use when you know the exact name of a quest, character, item, or
    location you need to look up. Preferred over semantic_search when
    traversing prerequisite chains.
    Falls back to semantic_search if the exact title is not found.
    """
```

### 6.3 Graph Definition

`agent/graph.py` defines the LangGraph state graph.

**Nodes:**

| Node | Function | Description |
| --- | --- | --- |
| `router` | `nodes.route_question` | Classifies question type; sets initial retrieval strategy |
| `retrieve` | `nodes.retrieve` | Calls semantic search or exact lookup based on current state |
| `extract` | `nodes.extract_info` | LLM reads retrieved chunks; extracts prerequisites or direct answer |
| `check_complete` | `nodes.check_complete` | Decides whether more retrieval is needed or answer is ready |
| `synthesize` | `nodes.synthesize_answer` | LLM composes final answer from all accumulated context |
| `handle_limit` | `nodes.handle_iteration_limit` | Returns partial answer when max iterations reached |

**Edges:**

```text
START → router
router → retrieve
retrieve → extract
extract → check_complete
check_complete → synthesize          [if needs_more_retrieval = False]
check_complete → handle_limit        [if iteration_count >= agent_max_iterations]
check_complete → retrieve            [if needs_more_retrieval = True]
synthesize → END
handle_limit → END
```

### 6.4 Node Implementations

`agent/nodes.py`

```python
def route_question(state: AgentState) -> AgentState:
    # classifies question as prerequisite-type or direct-answer-type
    # prerequisite-type: sets initial entity to resolve from question
    # direct-answer-type: sets semantic search as first action

def retrieve(state: AgentState) -> AgentState:
    # increments iteration_count
    # if entity in state.visited: skips (cycle prevention)
    # calls get_page() for known entities, semantic_search() otherwise
    # appends results to state.retrieved_context
    # adds resolved entity to state.visited

def extract_info(state: AgentState) -> AgentState:
    # LLM prompt: given these chunks, what are the prerequisites
    # for [entity]? Are there any unresolved prerequisites?
    # updates state.prerequisite_chain
    # sets state.needs_more_retrieval based on LLM response
    # adds any newly identified prerequisite entities to queue

def check_complete(state: AgentState) -> AgentState:
    # checks iteration_count against settings.agent_max_iterations
    # passes through state unchanged; routing handled by conditional edges

def synthesize_answer(state: AgentState) -> AgentState:
    # LLM prompt: given all retrieved context and the prerequisite
    # chain built so far, provide a complete answer to the original question
    # populates state.final_answer

def handle_iteration_limit(state: AgentState) -> AgentState:
    # LLM prompt: based on what has been found so far (partial chain),
    # provide the best available answer and note it may be incomplete
    # populates state.final_answer with partial result
```

### 6.5 MLflow Tracing

MLflow tracing is applied to the LangGraph agent using MLflow's LangChain/LangGraph autolog integration.

```python
import mlflow

mlflow.langchain.autolog(
    log_traces=True,        # logs tool calls, node inputs/outputs
    log_models=False        # model logging not needed for tracing
)
```

This is enabled at application startup. Each agent invocation produces a trace in MLflow that shows the sequence of nodes visited, tool calls made, and iteration count — visible in the MLflow UI.

---

## 7. Evaluation Pipeline

### 7.1 Dataset Format and Storage

The eval dataset is stored as a JSON file at `data/eval/golden_set.json` and versioned in git. Synthetic pairs are generated separately and merged before eval runs.

**Golden set maintenance** — the golden set must be reviewed periodically to catch questions whose expected answers have become stale due to game updates. This review should be triggered by any game patch or wiki sync that touches pages referenced by golden set questions. Stale entries should be updated in place and committed to git with a note in the commit message indicating which entries changed and why.

```json
[
  {
    "inputs": { "question": "What do I need to complete to unlock Ice and Glow?" },
    "expected_response": "To unlock Ice and Glow, you must first complete Straight to Your Heart and Absence Makes the Heart quest series.",
    "metadata": {
      "source": "golden",
      "question_type": "prerequisite"
    }
  }
]
```

### 7.2 Synthetic Generation

`eval/generate.py` generates Q&A pairs from wiki chunks using the LLM.

**System prompt:**

```text
You are generating evaluation data for a RAG system about the game
Hello Kitty Island Adventure. Given a wiki chunk and a question category,
generate question/answer pairs that a player might ask.

Rules:
- Every question must be answerable solely from the provided chunk.
  Do not use outside knowledge.
- Every answer must be a complete standalone response.
- Questions must be specific, not vague. Bad: "What is crafting?"
  Good: "What materials do I need to craft a Wooden Bench?"
- Do not generate questions about information not present in the chunk.
- Generate fewer pairs for short or sparse chunks (1 pair if the chunk
  contains only one distinct fact).
- Return only valid JSON. No preamble, no markdown fences.
```

**User prompt:**

```text
Chunk:
{chunk_text}

Question category: {question_type}

Generate {n} question/answer pairs. Return as JSON array:
[{{"question": "...", "answer": "...", "question_type": "..."}}]
```

**Generation process:**

```python
def generate_for_chunk(
    chunk: dict,
    question_type: str,
    n_pairs: int | None = None
) -> list[dict]:
    # n_pairs defaults to 2 for chunks > 200 tokens, 1 for shorter chunks
    # calls LLM, strips any markdown fences, parses JSON
    # validates each pair has question, answer, question_type fields
    # returns validated pairs or empty list on parse failure
```

### 7.3 MLflow Experiment Runner

`eval/runner.py` runs a single MLflow experiment with specified parameters.

```python
def run_experiment(
    experiment_name: str,
    run_name: str,
    params: dict,               # chunking_strategy, embedding_model, llm_model, top_k, etc.
    dataset_path: str
) -> mlflow.entities.Run:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        dataset = load_dataset(dataset_path)
        mlflow_dataset = mlflow.data.from_pandas(
            pd.DataFrame(dataset),
            name=f"hkia_eval_{version}"
        )
        mlflow.log_input(mlflow_dataset, context="eval")

        results = mlflow.genai.evaluate(
            data=pd.DataFrame(dataset),
            predict_fn=run_rag_pipeline,
            scorers=[
                mlflow.genai.scorer.answer_correctness(),
                mlflow.genai.scorer.faithfulness(),
                mlflow.genai.scorer.context_recall(),
                mlflow.genai.scorer.context_precision(),
            ]
        )

        # log per-question-type breakdowns
        for question_type in QUESTION_TYPES:
            subset = results[results["question_type"] == question_type]
            for metric in METRICS:
                mlflow.log_metric(
                    f"{metric}/{question_type}",
                    subset[metric].mean()
                )

        return run
```

### 7.4 Planned Experiments

Each experiment varies one parameter from the baseline. All other parameters are held constant.

**Baseline configuration:**

- Chunking: recursive, chunk_size=512, overlap=64
- Embedding: nomic-embed-text:v1.5 (Ollama)
- LLM: llama3 (Ollama)
- top_k: 5, similarity_threshold: 0.7

| Experiment | Parameter Changed | Values |
| --- | --- | --- |
| E1 — Chunking | chunking_strategy | recursive (baseline) vs section |
| E2 — Embedding | embedding_model | nomic-embed-text (baseline) vs text-embedding-3-small |
| E3 — LLM | llm_model | llama3 (baseline) vs gpt-4o vs claude-sonnet |
| E4 — Retrieval | top_k | 3, 5 (baseline), 10 |
| E4 — Retrieval | similarity_threshold | 0.6, 0.7 (baseline), 0.8 |

**Note on E2:** Switching embedding models requires building a new ChromaDB collection (`hkia_v2`) via full re-ingestion before running the experiment. See Section 5.2.

---

## 8. Gradio Frontend

### 8.1 Interface Layout

`app/gradio_app.py` implements a single-page Gradio chat interface.

```text
┌─────────────────────────────────────────────┐
│  🌸 Hello Kitty Island Adventure Assistant  │
├─────────────────────────────────────────────┤
│                                             │
│  [Chat history display]                     │
│                                             │
├─────────────────────────────────────────────┤
│  [Text input]                [Send button]  │
└─────────────────────────────────────────────┘
```

### 8.2 Implementation

```python
import gradio as gr
from agent.graph import build_graph

graph = build_graph()

def respond(message: str, history: list) -> str:
    state = AgentState(
        question=message,
        messages=history_to_messages(history)
    )
    result = graph.invoke(state)
    return result.final_answer

demo = gr.ChatInterface(
    fn=respond,
    title="🌸 Hello Kitty Island Adventure Assistant",
    description="Ask anything about HKIA — quests, characters, crafting, locations, and more.",
    examples=[
        "What quests do I need to complete to unlock Ice and Glow?",
        "What gifts does Keroppi like?",
        "How do I craft a Wooden Bench?",
        "How do I get to Icy Peak?",
        "How does the daily gift limit work?",
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
```

---

## 9. Sync Scheduler

### 9.1 CLI Entry Point

`sync.py` is the single entry point for all sync operations, usable both manually and via cron.

```python
# Usage:
#   python sync.py --mode full        # full re-ingest of all pages
#   python sync.py --mode incremental # default; only changed/new pages
#   python sync.py --mode status      # print ingestion state summary

import argparse
from ingestion.pipeline import run_full_ingestion, run_incremental_ingestion
from ingestion.state_db import get_status_summary

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["full", "incremental", "status"],
                    default="incremental")
args = parser.parse_args()

if args.mode == "full":
    run_full_ingestion()
elif args.mode == "incremental":
    run_incremental_ingestion()
elif args.mode == "status":
    print(get_status_summary())
```

### 9.2 Cron Setup

Add to crontab via `crontab -e`:

```bash
# Run incremental HKIA wiki sync every Sunday at 3am
0 3 * * 0 cd /path/to/hkia-rag && .venv/bin/python sync.py --mode incremental >> logs/sync.log 2>&1
```

The cron entry is documented in the README but not automated — it must be set up manually on the local machine. This is an acceptable tradeoff for a local-first application.

---

## 10. Testing Strategy

### 10.1 Framework Decision

**pytest** is used over `unittest` for the following reasons: cleaner function-based test syntax (no class inheritance required), richer assertion failure output, and `pytest-mock` support for mocking external API calls during ingestion tests. The extra dependency is acceptable given the project's existing footprint.

**Tradeoff acknowledged:** `unittest` is part of the Python standard library and requires no extra dependency. It is not chosen here because the developer experience difference is meaningful for a project this size, and pytest is the de facto standard in ML/data engineering contexts that technical reviewers will expect.

### 10.2 Scope

Tests are targeted at the three highest-risk logic areas. Full coverage is not a goal for v1 — the tradeoff is development speed against safety. The three areas chosen are those where silent failures would be hardest to detect through manual use of the app.

### 10.3 Test Specifications

**`tests/test_ingestion_idempotency.py`**

Verifies that re-running ingestion after a partial failure produces a clean state with no duplicate chunks.

```python
def test_rerun_after_failure_produces_no_duplicates(mocker):
    # mock MediaWiki API to return 3 pages
    # mock embedder to succeed for pages 1 and 2, raise on page 3
    # run ingestion → expect partial completion (pages 1,2 complete; 3 in_progress)
    # fix mock so all pages succeed
    # re-run ingestion
    # assert ChromaDB contains exactly 1 set of chunks per page (no duplicates)
    # assert all pages marked complete in SQLite

def test_status_transitions_correctly():
    # verify page moves: pending → in_progress → complete on success
    # verify page stays in_progress on failure
    # verify complete pages are skipped on re-run
```

**`tests/test_agent_cycle_detection.py`**

Verifies the agent does not loop infinitely when wiki pages form a cycle.

```python
def test_agent_stops_on_circular_prerequisites(mocker):
    # mock get_page tool to return:
    #   Quest A → prerequisite: Quest B
    #   Quest B → prerequisite: Quest A  (circular)
    # invoke agent with question about Quest A
    # assert agent terminates (does not exceed max_iterations)
    # assert final_answer is populated (partial answer returned)
    # assert visited set contains both Quest A and Quest B

def test_agent_respects_max_iterations(mocker):
    # mock get_page to always return a new unseen prerequisite
    # invoke agent
    # assert iteration_count == settings.agent_max_iterations
    # assert handle_iteration_limit node was reached
```

**`tests/test_embedding_version_guard.py`**

Verifies the embedding model mismatch guard fires correctly.

```python
def test_guard_raises_on_model_mismatch(mocker):
    # mock ChromaDB to return chunks with embedding_model = "nomic-embed-text:v1.5"
    # set settings.embedding_model = "text-embedding-3-small"
    # set settings.embedding_model_version = "3"
    # call verify_collection_embedding_model()
    # assert EmbeddingModelMismatchError is raised

def test_guard_passes_on_model_match(mocker):
    # mock ChromaDB to return chunks with embedding_model = "nomic-embed-text:v1.5"
    # set settings to match
    # assert no exception raised
```

---

## 11. Dependency Decisions Log

| Decision | Choice | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Package manager | uv | pip + venv, poetry, conda | Faster resolution, modern standard, near-identical workflow to pip |
| Ingestion state store | SQLite | PostgreSQL, JSON files, ChromaDB metadata | Zero infra, standard library, atomic writes, queryable; JSON rejected for fragility on partial writes |
| Test framework | pytest + pytest-mock | unittest | Cleaner syntax, richer failure output, mock support; extra dependency acceptable |
| Sync scheduling | cron + CLI | APScheduler | APScheduler requires running process; cron is simpler and more reliable for local scheduled jobs |
| Chunking utilities | LangChain text splitters | Custom implementation | Well-tested, handles edge cases; isolated utility use avoids framework lock-in |
| Agent framework | LangGraph | Custom loop, LangChain AgentExecutor | Stateful graph model fits multi-hop traversal; generalizes beyond quest prerequisites to characters and locations |
