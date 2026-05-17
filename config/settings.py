"""Central runtime configuration loaded from environment / ``.env``."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime knobs for ingestion, retrieval, and the agent.

    Loaded from ``.env`` at process start. Field defaults are the
    values the project ships with; override via ``.env`` or
    process-level env vars (``ENV_VAR_NAME=...``). Every read happens
    through the module-level ``settings`` singleton at the bottom of
    this file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # MediaWiki
    wiki_base_url: str = Field(
        default="https://hellokittyislandadventure.wiki.gg",
        description=(
            "Wiki base URL. Used to construct ``source_url`` in chunk "
            "metadata; not used for API calls (see wiki_api_url)."
        ),
    )
    wiki_api_url: str = Field(
        default="https://hellokittyislandadventure.wiki.gg/api.php",
        description="MediaWiki API endpoint queried by ingestion.api_client.",
    )
    wiki_request_delay_seconds: float = Field(
        default=0.75,
        description=(
            "Sleep applied before every wiki request to respect the "
            "wiki's rate limits. Applies on retries too."
        ),
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./chroma_data",
        description="On-disk path for the ChromaDB persistent store.",
    )
    chroma_collection_name: str = Field(
        default="hkia_nomic-embed-text_recursive_v2",
        description=(
            "Collection name. Convention: hkia_{embedding_model}_"
            "{chunking_strategy}_v{n}. Bump v{n} when changing "
            "embedding_model or chunking_strategy — the startup sync "
            "check raises CollectionConfigMismatchError otherwise."
        ),
    )

    # Embedding
    embedding_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Which embedding backend the ingestion pipeline uses.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description=(
            "Embedding model name. Stored alongside chunks; drift from "
            "this value is treated as a hard error at startup."
        ),
    )
    embedding_model_version: str = Field(
        default="v1.5",
        description=(
            "Logical version paired with embedding_model into the "
            "``{model}:{version}`` identifier mirrored across SQLite "
            "and ChromaDB metadata."
        ),
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Required when embedding_provider='openai' or "
            "llm_provider='openai'. Also required by the eval pipeline's "
            "MLflow LLM judges, regardless of agent provider."
        ),
    )
    openai_embedding_batch_size: int = Field(
        default=100,
        description="Number of chunks per OpenAI /embeddings request.",
    )

    # LLM
    llm_provider: Literal["ollama", "openai", "anthropic"] = Field(
        default="openai",
        description="Which LLM backend the agent calls for extract / synthesize.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name passed verbatim to the configured provider.",
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Required when llm_provider='anthropic'.",
    )
    ollama_request_timeout_seconds: int = Field(
        default=180,
        description=(
            "Per-request timeout (seconds) for Ollama chat calls. "
            "Default suits small/medium models; raise for 70B+ local "
            "models where a single completion may legitimately exceed "
            "three minutes."
        ),
    )

    # LangGraph agent
    agent_max_iterations: int = Field(
        default=10,
        description=(
            "Hard ceiling on the agent's retrieve→extract loop. Only "
            "guaranteed termination path when extract repeatedly fails "
            "to parse JSON — do not raise without understanding the "
            "iteration × parse-retry budget."
        ),
    )

    # SQLite
    state_db_path: str = Field(
        default="./data/ingestion_state.db",
        description="SQLite file holding per-page ingestion status and revision IDs.",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="./mlruns",
        description=(
            "MLflow tracking backend. Local path or remote URI; the "
            "runner reads this when starting an experiment."
        ),
    )
    mlflow_experiment_name: str = Field(
        default="hkia_rag",
        description="Default MLflow experiment name for agent traces.",
    )

    # Chunking
    chunking_strategy: Literal["recursive", "section"] = Field(
        default="recursive",
        description=(
            "Chunker strategy. 'section' preserves wiki section "
            "headings as retrieval context; changing requires bumping "
            "chroma_collection_name and re-ingesting."
        ),
    )
    chunk_size: int = Field(
        default=512,
        description="Target maximum character count per chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Overlap in characters between consecutive chunks.",
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks returned by semantic_search per call.",
    )

    # Retrieval logging
    retrieval_log_enabled: bool = Field(
        default=True,
        description=(
            "Toggle for the per-event JSONL retrieval logger. Disable "
            "in tests or when reducing on-disk PII surface."
        ),
    )
    retrieval_log_file: str = Field(
        default="logs/retrieval.jsonl",
        description="Destination file for retrieval-event JSONL.",
    )


settings = Settings()
