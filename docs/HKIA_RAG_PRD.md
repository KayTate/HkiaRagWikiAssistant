# Product Requirements Document

## Hello Kitty Island Adventure — RAG Q&A Application

**Version:** 1.0  
**Status:** Draft  
**Author:** Kaycee  
**Date:** April 2026

---

## 1. Overview

### 1.1 Purpose

This document defines the requirements for a Retrieval-Augmented Generation (RAG) application that answers natural language questions about Hello Kitty Island Adventure (HKIA) using the official HKIA wiki as its knowledge base. The application serves two purposes: a functional tool for personal use while playing the game, and a portfolio artifact demonstrating production-grade MLOps practices including systematic evaluation via MLflow.

### 1.2 Background

The HKIA wiki at `hellokittyislandadventure.wiki.gg` contains approximately 5,249 articles covering quests, characters, items, crafting, locations, and game mechanics. Answering some questions — particularly those involving unlock prerequisites for quests, characters, and locations — requires traversing chains of dependencies that are distributed across multiple wiki pages. A standard single-pass RAG pipeline cannot handle this reliably. The application addresses this with an agentic retrieval architecture using LangGraph.

---

## 2. Goals

- Answer natural language questions about HKIA accurately across five question categories
- Handle multi-hop prerequisite questions for quests, characters, and locations via agentic retrieval
- Demonstrate systematic RAG evaluation using MLflow experiment tracking with RAGAS-style metrics
- Keep the system runnable locally with a shareable demo

### 2.1 Non-Goals (v1)

- Multi-user support or authentication
- Real-time wiki sync (scheduled refresh is sufficient)
- Mobile-optimized UI
- Support for wikis other than HKIA

---

## 3. Users

| Audience | Primary Need |
| --- | --- |
| Kaycee (personal use) | Fast, accurate answers while playing HKIA |
| Recruiters / technical reviewers | Evidence of MLOps, RAG, and evaluation pipeline skills |
| General RAG demo audience | Demonstration of agentic RAG on a real-world corpus |

---

## 4. Functional Requirements

### 4.1 Question Categories

The application must handle all five of the following question types at launch:

**Quest prerequisites / unlock chains**
Multi-hop questions like "what do I need to complete to unlock Ice and Glow?" The agent must traverse prerequisite chains of arbitrary depth until it reaches quests with no dependencies.

**General game mechanics**
Single-hop questions about how systems work: stamina, gifting, friendship levels, tools, daily reset behavior, etc.

**Character friendship info**
Questions about specific characters: what gifts they like, what friendship levels unlock, companion abilities, and related quests.

**Item and crafting info**
Questions about materials, crafting recipes, item sources, and what items are used for.

**Location and exploration**
Questions about where things are, how to reach areas, and what unlock prerequisites exist for locations (which may not be explicitly labeled as "prerequisites" in the wiki).

### 4.2 Interaction Model

- Simple chat interface: user types a natural language question, receives a natural language answer
- No structured filters or categories exposed to the user — the agent determines question type internally
- Conversation history maintained within a session; no cross-session memory required

### 4.3 Agentic Retrieval (LangGraph)

The application uses a LangGraph agent rather than a single-pass RAG pipeline. The agent has access to two retrieval tools:

**Semantic search tool** — queries ChromaDB using vector similarity; used for broad questions where the relevant page title is not known in advance.

**Exact page lookup tool** — fetches a specific wiki page by title; used when the agent has identified a specific entity (quest name, character name, location name) it needs to resolve. This is the primary tool for prerequisite chain traversal.

The agent state tracks: the original question, entities resolved so far, the prerequisite chain built so far, and a visited set for cycle prevention. The agent loops until it determines it has sufficient information to answer, then synthesizes a final response.

**Failure modes and safeguards** — the agent must enforce a maximum iteration limit (e.g. 10 hops) to prevent infinite loops on malformed or circular wiki content. If the limit is reached, the agent returns a partial answer based on what it has resolved so far rather than erroring or hanging. Page title mismatches (e.g. a prerequisite name that doesn't exactly match a wiki page title) are handled gracefully by falling back to semantic search before giving up.

---

## 5. Data Ingestion Pipeline

### 5.1 Initial Ingestion

1. Enumerate all pages via the MediaWiki API (`action=query&list=allpages`)
2. Fetch wikitext per page (`action=parse&prop=wikitext`)
3. Strip wikitext to plain text using `mwparserfromhell`
4. Chunk text (see Section 5.3)
5. Embed chunks and store in ChromaDB with metadata
6. Record revision ID, timestamp, and ingestion status per page for incremental sync and idempotency

### 5.2 Incremental Sync

- Scheduled weekly refresh
- For each page, compare stored revision ID against current revision via the API
- If changed: delete all existing chunks for that page from ChromaDB, re-ingest
- If new page: ingest normally
- Deleted pages: flagged for manual review in v1; automated deletion in v2

### 5.3 Ingestion Idempotency

The ingestion script must be safe to re-run after a partial failure without producing duplicate or inconsistent state. Each page tracks one of three ingestion statuses: `pending`, `in_progress`, or `complete`. On retry, the script skips pages marked `complete` and reprocesses anything in `pending` or `in_progress`. Before inserting chunks for a page, any existing chunks for that page are deleted from ChromaDB first, ensuring a clean state regardless of where a previous run failed.

### 5.4 Rate Limiting

**MediaWiki API** — the ingestion script enforces a small sleep between requests (e.g. 0.5–1s) to avoid hammering the wiki server. This is good practice regardless of scale.

**OpenAI Embeddings API** — used during the embedding model comparison experiment. Chunks must be submitted in batches with retry logic on rate limit errors (HTTP 429) using exponential backoff. Embedding chunks one at a time is not acceptable for a corpus of this size.

### 5.5 Chunking Strategy

The chunking strategy is a tunable parameter for MLflow experiments. Two strategies are implemented:

- **Recursive character splitting** — baseline; splits on paragraph → sentence → character boundaries
- **Section-based splitting** — splits on MediaWiki section headings (`==Section==`); preserves semantic coherence for wiki-structured content

### 5.6 Chunk Metadata

Every chunk is stored with the following metadata in ChromaDB:

| Field | Description |
| --- | --- |
| `source_title` | Wiki page title (used for delete-by-source on update) |
| `source_url` | Full URL to the wiki page |
| `section` | Section heading the chunk came from (if applicable) |
| `category` | Wiki category (e.g. Quests, Characters, Items) |
| `chunk_index` | Position of chunk within the source page |
| `revision_id` | MediaWiki revision ID at time of ingestion |
| `ingested_at` | Timestamp of ingestion |
| `embedding_model` | Name and version of the embedding model used (e.g. `nomic-embed-text:v1.5`) |

**Embedding model versioning** — the `embedding_model` metadata field is critical for collection integrity. Vectors produced by different embedding models occupy incompatible spaces and cannot be mixed in the same collection. Any change to the embedding model (version upgrade or model swap for an experiment) requires a full re-ingest into a new versioned ChromaDB collection (e.g. `hkia_v1`, `hkia_v2`). The application config specifies which collection name to query at runtime, allowing atomic cutover from one collection to another without downtime.

---

## 6. Evaluation Pipeline

### 6.1 Eval Dataset

The eval dataset is composed of two source types:

**Synthetic (LLM-generated)**
Wiki chunks are fed to an LLM with a prompt requesting question/answer pairs. Covers the four general knowledge question types: game mechanics, character friendship, crafting, and location. Spot-checked for quality before inclusion.

The generation prompt must include:

- **The chunk text** — grounds all questions in actually retrievable content
- **The question category** — one of the four general knowledge question types, specified explicitly to ensure balanced coverage across categories
- **Pair count instruction** — 2–3 pairs per chunk by default; the LLM is instructed to generate fewer pairs for sparse chunks rather than padding with low-quality questions
- **Answerability constraint** — each question must be answerable solely from the provided chunk without requiring outside knowledge
- **Completeness constraint** — each answer must be a complete standalone response, not a fragment or forward reference
- **Specificity instruction** — questions must be specific (e.g. "what materials do I need to craft a wooden bench?") rather than vague (e.g. "what is crafting?")
- **Negative constraint** — the LLM must not generate questions referencing information absent from the chunk, preventing hallucinated Q&A pairs
- **Strict JSON output format** — schema enforced with `question`, `answer`, and `question_type` fields for reliable parsing without cleanup

**Golden set (hand-written)**
Manually authored questions written from domain knowledge. Covers all five question types, with particular emphasis on prerequisite chain questions (quests, characters, locations) since these are not generated synthetically. Targets approximately 8–10 questions per question type for roughly 40–50 total, providing meaningful coverage now that the golden set carries the full prerequisite category.

Each row in the dataset has the shape:

```json
{
  "inputs": { "question": "..." },
  "expected_response": "...",
  "metadata": {
    "source": "synthetic | golden",
    "question_type": "prerequisite | mechanic | friendship | crafting | location"
  }
}
```

The dataset is version-controlled in git and logged as an MLflow dataset artifact on each eval run.

### 6.2 Agent Observability

MLflow tracing is used to instrument the LangGraph agent. Each agent run logs the sequence of tool calls made, the inputs and outputs of each node, and the total number of retrieval hops. Traces are stored in MLflow alongside experiment runs, making it possible to inspect exactly what the agent did for any given query. This is surfaced in the MLflow UI and serves as a key demo artifact for the portfolio.

### 6.3 MLflow Experiments

Four experiments are planned, each varying one component while holding others constant:

| Experiment | Variable | Baseline | Alternatives |
| --- | --- | --- | --- |
| Chunking | Chunking strategy | Recursive character splitting | Section-based splitting |
| Embedding | Embedding model | `nomic-embed-text` (Ollama) | `text-embedding-3-small` (OpenAI) |
| LLM | Generation model | Local Ollama model (e.g. llama3) | Claude or GPT-4o via API |
| Retrieval | top-k, similarity threshold | k=5, threshold=0.7 | Grid search over k∈{3,5,10}, threshold∈{0.6,0.7,0.8} |

### 6.4 Metrics

RAGAS-style metrics tracked per run:

- **Answer correctness** — semantic similarity of generated answer to expected response
- **Faithfulness** — whether the answer is grounded in the retrieved context
- **Context recall** — whether the relevant chunks were retrieved
- **Context precision** — proportion of retrieved chunks that were actually relevant

Metrics are additionally sliced by `question_type` and `source` metadata to surface per-category performance.

---

## 7. Tech Stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| Wiki ingestion | MediaWiki API + `mwparserfromhell` | Native API access; clean wikitext parsing |
| Vector store | ChromaDB (local) | Zero infra, persistent on disk, Python-native |
| Embeddings | `nomic-embed-text` via Ollama (baseline) | Already installed; free; swappable |
| Agentic orchestration | LangGraph | Stateful multi-hop retrieval; generalizes to all entity types |
| LangChain utilities | Text splitters only | Avoids framework lock-in while reusing tested components |
| LLM | Local Ollama model (baseline) | Free; swappable via MLflow experiments |
| Frontend | Gradio | Minimal code; standard for ML demos; fast to ship |
| Experiment tracking | MLflow | Target platform; GenAI eval API |
| Version control | Git | Code + eval dataset |

---

## 8. Infrastructure & Data Protection

### 8.1 Runtime Environment

The application runs locally. No cloud deployment is required for v1. A demo recording or live screen share is the primary sharing mechanism for recruiters.

### 8.2 Data Protection

| Artifact | Treatment |
| --- | --- |
| Pipeline code | Git (source of truth) |
| Eval dataset | Git (versioned alongside code) |
| ChromaDB persist directory | Periodic snapshot (directory copy); treated as reproducible, not irreplaceable |
| MLflow `mlruns` directory | Periodic snapshot; represents non-reproducible experiment history |

The ChromaDB vector store is treated as a derived artifact — it can be fully rebuilt from the wiki via the ingestion pipeline if lost. The MLflow tracking store is the higher-priority backup target.

---

## 9. Known Limitations (v1)

- **Stale data between sync cycles** — the agent answers based on whatever is currently in ChromaDB; wiki updates between scheduled syncs are not reflected
- **Prerequisite coverage varies by entity type** — quest prerequisites are explicitly labeled in infoboxes; character and location prerequisites may be described in prose and are more dependent on retrieval quality
- **No cross-session memory** — conversation history resets between sessions
- **Deleted wiki pages** — not automatically removed from ChromaDB in v1

---

## 10. Future Improvements (v2+)

- Automated deleted-page cleanup in incremental sync
- Deployment to HuggingFace Spaces or Fly.io for shareable public URL
- Webhook or more frequent sync for near-real-time wiki updates
- Expanded LangGraph toolset (e.g. category browsing tool)
- Cross-session conversation memory
- Force sync command triggerable by the user (e.g. Gradio button or CLI command) to pull latest wiki content outside the weekly schedule
- Remote MLflow tracking server to persist experiment history independently of the local machine, enabling metric collection if the app is shared or deployed
- Expose agent trace details in the Gradio UI — show the user which tool calls were made and what the agent's reasoning steps were for a given answer
