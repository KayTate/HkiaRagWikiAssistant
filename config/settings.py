from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MediaWiki
    wiki_base_url: str = "https://hellokittyislandadventure.wiki.gg"
    wiki_api_url: str = "https://hellokittyislandadventure.wiki.gg/api.php"
    wiki_request_delay_seconds: float = 0.75

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "hkia_nomic-embed-text_recursive_v1"

    # Embedding
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_model_version: str = "v1.5"
    openai_api_key: str = ""
    openai_embedding_batch_size: int = 100

    # LLM
    llm_provider: Literal["ollama", "openai", "anthropic"] = "openai"
    llm_model: str = "gpt-5.4-mini"
    anthropic_api_key: str = ""

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

    # Retrieval logging
    retrieval_log_enabled: bool = True
    retrieval_log_file: str = "logs/retrieval.jsonl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
