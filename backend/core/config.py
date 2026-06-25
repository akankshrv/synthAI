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
    rrf_top_n: int = 40
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval feature flags (overridable by eval CLI)
    enable_bm25: bool = True
    enable_rerank: bool = True
    enable_cache: bool = True
    enable_mmr: bool = True
    enable_dedup: bool = True
    enable_query_decomposition: bool = True
    dedup_similarity_threshold: float = 0.92
    mmr_lambda: float = 0.7
    max_decomposed_queries: int = 3
    max_search_urls: int = 8

    # Persistent storage
    chroma_path: str = "./chroma_data"
    chroma_collection_name: str = "web_chunks"
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "synthai"
    url_cache_ttl_hours: float = 24.0
    chroma_ttl_hours: float = 72.0

    # Chunking (token-aware)
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Query rewriting / conversation
    enable_query_rewrite: bool = True
    enable_query_contextualization: bool = True
    max_history_turns: int = 6
    query_rewrite_model: str = "openai/gpt-oss-20b:free"
    query_rewrite_timeout: float = 15.0

    # HTTP timeouts (seconds)
    http_timeout: float = 30.0
    llm_timeout: float = 120.0

    # Observability / eval
    enable_debug_events: bool = True
    trace_log_path: str = "traces.jsonl"
    eval_baseline_dir: str = "eval/baselines"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
