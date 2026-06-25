import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.config import settings


class PipelineTracer:
    def __init__(self, original_query: str) -> None:
        self.original_query = original_query
        self.rewritten_query: str | None = None
        self.stages: dict[str, float] = {}
        self.retrieved_chunks: list[dict[str, Any]] = []
        self.prompt_token_count: int | None = None
        self.completion_token_count: int = 0
        self.cache_stats: dict[str, Any] = {}
        self._stage_starts: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 2)

    def record_chunks(self, chunks: list[dict]) -> None:
        self.retrieved_chunks = chunks

    def record_cache_stats(self, stats: dict) -> None:
        self.cache_stats = stats

    def record_prompt_tokens(self, count: int) -> None:
        self.prompt_token_count = count

    def add_completion_tokens(self, count: int) -> None:
        self.completion_token_count += count

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "stages_ms": self.stages,
            "retrieved_chunks": self.retrieved_chunks,
            "cache_stats": self.cache_stats,
            "prompt_token_count": self.prompt_token_count,
            "completion_token_count": self.completion_token_count,
            "total_latency_ms": round(sum(self.stages.values()), 2),
        }

    def summary(self) -> dict[str, Any]:
        data = self.to_dict()
        return {
            "original_query": data["original_query"],
            "rewritten_query": data["rewritten_query"],
            "stages_ms": data["stages_ms"],
            "cache_stats": data.get("cache_stats", {}),
            "sub_queries": data.get("cache_stats", {}).get("sub_queries", []),
            "chunk_count": len(data["retrieved_chunks"]),
            "top_chunks": [
                {
                    "id": c.get("id"),
                    "source": c.get("source"),
                    "dense_score": c.get("dense_score"),
                    "bm25_score": c.get("bm25_score"),
                    "rrf_score": c.get("rrf_score"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in data["retrieved_chunks"]
            ],
            "prompt_token_count": data["prompt_token_count"],
            "completion_token_count": data["completion_token_count"],
            "total_latency_ms": data["total_latency_ms"],
        }

    def flush(self) -> None:
        log_path = Path(settings.trace_log_path)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
