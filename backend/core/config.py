from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    tavily_api_key: str = ""
    openrouter_api_key: str = ""
    jina_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-120b:free"
    cors_origins: str = "http://localhost:3000"

    # Retrieval tunables
    top_k: int = 8
    retrieval_candidate_k: int = 20
    rrf_k: int = 60
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking (~400-600 tokens with ~50 token overlap)
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Query rewriting
    enable_query_rewrite: bool = True
    query_rewrite_model: str = "openai/gpt-oss-20b:free"
    query_rewrite_timeout: float = 15.0

    # HTTP timeouts (seconds)
    http_timeout: float = 30.0
    llm_timeout: float = 120.0

    # Observability
    enable_debug_events: bool = True
    trace_log_path: str = "traces.jsonl"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
