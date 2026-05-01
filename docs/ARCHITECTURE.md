# HKIA RAG — Architecture

This is the living description of the system as it stands today: how
the subsystems wire together, where to look for what, and which
invariants are load-bearing. When this document and the source
disagree, the source wins — but please update this document in the
same change so the next reader is not misled.

## Reading guide

| Document | Scope |
| --- | --- |
| `README.md` | Setup, day-to-day operator commands, environment variables |
| `docs/ARCHITECTURE.md` (this file) | Current architecture, invariants, where things live |
| `docs/HKIA_RAG_PRD.md` | Original product requirements |
| `docs/HKIA_RAG_TDD.md` | v1 technical design (snapshot; implementation has evolved) |
| `config/settings.py` | Canonical list of runtime knobs with `Field(description=...)` |
| `pyproject.toml` | Canonical dependency pins |

## System layout

Three loosely-coupled subsystems wired together by `config/settings.py`
(pydantic-settings, reads `.env`):

```text
       ┌─────────────────┐
       │ MediaWiki API   │
       └────────┬────────┘
                │  (rate-limited batch fetch)
                ▼
   ┌────────────────────────┐         ┌─────────────────┐
   │  Ingestion subsystem   │────────▶│  Vector store   │
   │  ingestion/  sync.py   │  upsert │  vectorstore/   │◀──┐
   └───────────┬────────────┘         │  (ChromaDB)     │   │
               │ status                └─────────────────┘   │
               ▼                                             │
   ┌────────────────────────┐                  ┌─────────────┴────┐
   │  SQLite state DB       │                  │  Agent subsystem │
   │  ingestion/state_db.py │                  │  agent/          │
   └────────────────────────┘                  │  (LangGraph)     │
                                               └────────┬─────────┘
                                                        │
                                       ┌────────────────┼─────────────┐
                                       ▼                              ▼
                            ┌──────────────────────┐      ┌──────────────────┐
                            │  Gradio frontend     │      │  Eval subsystem  │
                            │  app/gradio_app.py   │      │  eval/           │
                            └──────────────────────┘      │  (MLflow judges) │
                                                          └──────────────────┘
```

Cross-cutting modules live in `common/` (currently just shared HTTP
retry predicates) and `config/` (settings + logging setup).

---

## 1. Ingestion subsystem (`ingestion/` + `sync.py`)

**Entry points:** `ingestion.pipeline.run_full_ingestion` /
`run_incremental_ingestion`. Both go through `run_startup_sync_check`
first (see "Embedding-model drift" below) and then process pending
pages.

**Per-page flow:**

```text
api_client          parser              chunker              embedder            vectorstore
──────────          ──────              ───────              ────────            ───────────
get_pages_wikitext  parse_wikitext      chunk_text           embed_chunks        upsert_chunks
_batch              extract_sections    (recursive |         (Ollama |           (delete_chunks
                    (template            section)             OpenAI)             _by_source first)
                     expansion)
```

State tracking lives in SQLite at `data/ingestion_state.db`
(`ingestion/state_db.py`). Each page row stores its `revision_id` and
the `embedding_model:version` it was embedded with. Incremental sync
compares revision IDs against the wiki to skip unchanged pages.

`upsert_chunks` is idempotent because chunk IDs are
`{page_title}::{chunk_index}`. The pipeline calls
`delete_chunks_by_source` before each upsert so a page whose chunk
count shrinks does not leave orphaned chunks behind.

### Per-page failure isolation

`_process_pending_pages` catches per-page exceptions so one bad page
does not halt the whole ingest run. The failing page keeps its
`in_progress` status and is retried on the next run; the run logs
`"Ingestion complete: N pages succeeded, M failed"` at the end.
Batch-level errors (wikitext fetch failures, embedding-model
mismatch) still propagate, since those signal a wiki- or
configuration-wide problem where continuing would just produce more
of the same.

### Bulk SQLite writes

The mark-pending phase uses `state_db.get_pages(titles)` and
`state_db.upsert_pages(rows)` — single SELECT and one `executemany`,
not N round-trips per page. For a wiki with thousands of pages this
matters: previously the mark phase opened a fresh SQLite connection
per page.

### Embedding-model drift

This is treated as a hard error and **is not auto-repaired.**
`pipeline.run_startup_sync_check` does two things on every ingest run:

1. `vectorstore_client.verify_collection_embedding_model()` samples
   up to 10 ChromaDB chunks (uniformly random — not insertion order,
   so a partial-reingest gap is detectable). If any sample was
   embedded with a different `embedding_model:embedding_model_version`
   than current settings, it raises `EmbeddingModelMismatchError`.
2. SQLite rows with a stale embedding model are silently reset to
   `pending` (with `revision_id = -1` sentinel) so they get
   re-fetched and re-embedded next run.

Operator remediation when (1) fires: bump `chroma_collection_name`
using the convention `hkia_{embedding_model}_{chunking_strategy}_v{n}`,
run `python sync.py --mode full`, then point your `.env` at the new
collection. The error message includes these steps.

---

## 2. Vector store subsystem (`vectorstore/`)

ChromaDB persistent client wrapped as a module-level singleton in
`vectorstore.client._chroma_client`. The collection name comes from
`settings.chroma_collection_name`.

**Public API:**

| Function | Purpose |
| --- | --- |
| `semantic_search(query_embedding, top_k, where=None)` | Vector search |
| `get_page_by_title(page_title)` | All chunks for one page, sorted by `chunk_index` |
| `upsert_chunks(page_title, chunks, embeddings, metadatas)` | Idempotent insert/replace |
| `delete_chunks_by_source(page_title)` | Remove all chunks for a page |
| `verify_collection_embedding_model()` | Drift guard (see ingestion) |
| `reset_client()` | Test-teardown helper to clear the singleton |

**Schema invariants** (`vectorstore/schema.py`):

`ChunkMetadata` carries `Field(...)` constraints — `chunk_index >= 0`,
`chunk_size > 0`, `chunk_overlap >= 0`, `revision_id >= -1` (the -1
sentinel is the stale-row reset value used by the startup sync check).
A negative or zero value would be a programming error and is rejected
at construction.

**Length parity at the upsert boundary:** `upsert_chunks` raises
`ValueError` immediately if `len(chunks) == len(embeddings) ==
len(metadatas)` does not hold. ChromaDB would otherwise raise a less
helpful error mid-write, leaving the collection in an indeterminate
state.

**Defensive None handling at the read boundary:** `get_page_by_title`
sorts by `(meta or {}).get("chunk_index", 0)` so a single legacy row
with `None` metadata cannot AttributeError the entire page-load path.

---

## 3. Agent subsystem (`agent/`)

LangGraph state graph defined in `agent/graph.py`. Nodes:

```text
router → retrieve → extract → check_complete → (retrieve | synthesize | handle_limit) → END
```

State (`agent/state.py`) accumulates `retrieved_context`,
`resolved_entities`, `prerequisite_chain`, and a `visited` set used
for cycle prevention during multi-hop prerequisite resolution.

### Routing and termination

`_route_after_check` priority order (`agent/graph.py`):

1. Iteration limit reached → `handle_iteration_limit`
2. No more retrieval needed → `synthesize_answer`
3. More retrieval needed → `retrieve`

**The iteration-limit branch MUST be checked first.** It is the only
guaranteed termination path when extract repeatedly fails to parse
JSON: `_handle_parse_failure` flips `needs_more_retrieval` back to
True after every retrieve, so without the iteration cap firing first
the graph would loop until LangGraph's internal recursion guard trips.

The cap is `settings.agent_max_iterations` (default 10). Combined
with `_extract_with_retry`'s 3-attempt JSON-parse budget, a worst-case
question runs `agent_max_iterations × 3 = 30` extract LLM calls plus
the synthesize call. That is intentional — JSON-retry attempts are
cheaper to lose than iterations, which sacrifice multi-hop reasoning.

### MLflow autolog timing

`agent.graph._enable_mlflow_autolog` is called inside `compile_graph`,
**not at module import**. This is deliberate so tests can patch
`mlflow` before compilation. Don't move it back to module-level
import.

### Module organisation

The `agent/` package is split for testability and separation of
concerns:

| Module | Responsibility |
| --- | --- |
| `graph.py` | StateGraph wiring + `compile_graph` |
| `state.py` | `AgentState` dataclass |
| `nodes.py` | Graph node functions; observability wrappers; stateful extraction orchestration |
| `llm.py` | LLM provider clients (Ollama, OpenAI, Anthropic) + tenacity retries |
| `extraction.py` | Pure text helpers (entity extraction, fence stripping) |
| `retrieval.py` | Entity → chunks resolution (title variants, opensearch, redirect-follow, semantic fallback) |
| `prompts.py` | System prompt strings used by the LLM-driven nodes |

The split between `extraction.py` and `nodes.py` is invariant-driven:
`extraction.py` is pure (text in, text out, no `AgentState`, no
logging, no LLM calls). Stateful orchestration (the JSON-parse retry
loop, the apply/handle helpers) stays in `nodes.py` because it
depends on `AgentState` and the observability wrappers.

### Test patching seams

Tests patch `agent.nodes._call_llm` rather than `agent.llm._call_llm`
because `nodes.py` imports the function and that is the binding the
production call sites resolve. Similarly, `agent.retrieval` is the
binding for `vs_get_page_by_title`, `vs_semantic_search`,
`embed_chunks`, and `_resolve_title_via_opensearch`.

### Observability

Every node emits structured JSONL events to the `retrieval` logger
(see Cross-cutting → Logging below). Each event carries a millisecond
UTC timestamp, the per-question `trace_id`, and the iteration count.
MLflow span instrumentation lives at the retrieval boundary in
`agent/retrieval.py` so the trace records exactly the chunks the LLM
will see — redirect-follow and semantic-search fallbacks are
implementation details of one logical retrieval and do not produce
sibling retriever spans.

---

## 4. Eval subsystem (`eval/` + `scripts/run_eval.py`)

`eval.runner.run_experiment` orchestrates MLflow experiments:

1. Compiles the agent graph.
2. Runs each golden-set question through it.
3. Scores responses with four `mlflow.genai` LLM judges
   (`Correctness`, `RelevanceToQuery`, `Summarization`,
   `RetrievalGroundedness`) using `openai:/gpt-4o-mini` as the judge
   model.
4. Logs aggregate scores plus per-question-type breakdowns
   (prerequisite, crafting, friendship, mechanic, location, general).

`eval.dataset` validates golden-set entries; `eval.generate` produces
synthetic Q&A pairs from wiki chunks (Ollama-only; cloud providers
are explicitly rejected to avoid surprise spend during dataset
generation).

**OpenAI key bridge:** `_ensure_openai_key_in_environ` copies
`settings.openai_api_key.get_secret_value()` into
`os.environ["OPENAI_API_KEY"]` before any judge runs, because the
MLflow LLM judges call the OpenAI SDK directly and only consult
process-level env vars. This means `OPENAI_API_KEY` is required for
eval runs even when the agent itself runs on Ollama or Anthropic.

---

## Cross-cutting

### HTTP retry predicates (`common/http.py`)

Single source of truth for "what counts as a retryable HTTP error":

- `is_transient_http_error(exc)` — matches `HTTPError` whose
  `response.status_code` is in `{429, 500, 502, 503, 504}`. Used by
  the cloud LLM clients (OpenAI, Anthropic) and the OpenAI embedder.
- `should_retry_request(exc)` — extends the above with
  `requests.ReadTimeout`. Used by the wiki API client because batch
  fetches pull large payloads where a slow response is almost always
  transient.

Permanent failures (401, 403, 404) propagate immediately rather than
burning the full backoff budget. The two predicates remain separate
because cloud LLM clients were not configured to retry on
`ReadTimeout` previously, and widening that behavior was kept out of
scope for the dedup that introduced this module.

### Secrets handling

API keys (`openai_api_key`, `anthropic_api_key`) are typed `SecretStr`
in `config/settings.py` so a stray `repr(settings)` in a log cannot
reveal the value. Production paths must call `.get_secret_value()` to
extract the raw string for headers and env-var assignments. The four
load-bearing call sites are in `agent/llm.py`, `ingestion/embedder.py`,
and `eval/runner.py`.

### Configuration

Every runtime knob lives in `config/settings.py` declared via
`pydantic.Field(default=..., description=...)`. The descriptions are
the canonical reference; `.env.example` is hand-maintained but
mirrors them. The `Settings` class uses
`model_config = SettingsConfigDict(env_file=".env", ...)` (pydantic
v2) — not the deprecated nested `class Config:` form.

### Logging

`config.logging_config.setup_logging` configures two destinations:

1. The root logger writes human-readable text to `logs/hkia.log` (and
   stdout) via a rotating file handler.
2. A dedicated `retrieval` logger writes one JSON object per line to
   `logs/retrieval.jsonl` (also rotating). The retrieval logger does
   not propagate to root, so structured events stay out of
   `hkia.log`.

`agent.nodes._log_event` is the only producer of retrieval events
and is gated by `settings.retrieval_log_enabled`. **Caution:** these
events include the full system prompt and user message; they are PII
at rest if the wiki ever serves a multi-user workload.

---

## Key invariants

A condensed list of things that, if you change them, will break the
system in a non-obvious way. Each is also documented at its source.

| Invariant | Where enforced |
| --- | --- |
| Embedding model in ChromaDB == `settings.embedding_model:_version`; mismatch is a hard error | `vectorstore.client.verify_collection_embedding_model` |
| `_route_after_check` checks the iteration cap **first** | `agent/graph.py:_route_after_check` |
| MLflow autolog runs at `compile_graph` time, not module import | `agent/graph.py:compile_graph` |
| `upsert_chunks` lengths must agree (chunks = embeddings = metadatas) | `vectorstore/client.py:upsert_chunks` |
| Per-page ingest failure must not halt the batch | `ingestion/pipeline.py:_process_pending_pages` |
| API keys are `SecretStr`; raw access is via `.get_secret_value()` | `config/settings.py` + four call sites |
| `revision_id = -1` is the stale-row sentinel | `ingestion/state_db.py` and `ChunkMetadata` |
| The `agent_max_iterations` cap is the **only** guaranteed termination on persistent JSON-parse failures | `agent/graph.py` + `agent/nodes.py` |

---

## Testing approach

160 tests across 17 files. Each test file pins a specific behavior
that, if it regressed, would silently degrade some end-user-facing
property of the system. The most load-bearing files:

| File | What it locks in |
| --- | --- |
| `test_agent_cycle_detection.py` | Agent terminates on circular prerequisites and respects iteration cap |
| `test_embedding_version_guard.py` | Drift guard fires on stale chunks anywhere in the collection |
| `test_ingestion_idempotency.py` | Re-ingestion after partial failure produces no duplicates; failures don't halt the batch |
| `test_http_retries.py` | Predicates filter transient vs permanent codes; 401 doesn't burn 5 attempts |
| `test_state_db.py` | Bulk helpers use exactly one connection regardless of batch size |
| `test_vectorstore.py` | Schema constraints reject bad metadata; upsert_chunks rejects length mismatches |
| `test_parser.py` | Every wiki template family expands to the documented form |
| `test_chunker.py` | Section strategy prepends heading to every sub-chunk |
| `test_embedder.py` | OpenAI batch results are sorted by `index` so vectors stay aligned with chunks |

Run with `pytest tests/`. Type-check strictly with `mypy .`. Lint with
`ruff check .`.
