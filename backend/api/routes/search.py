import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from models.schemas import SearchRequest
from services.llm import build_prompt, count_tokens, stream_llm_response
from services.pipeline import run_retrieval_stage
from services.tracer import PipelineTracer

router = APIRouter()


@router.post("/search")
async def search(body: SearchRequest) -> EventSourceResponse:
    if not settings.tavily_api_key or not settings.openrouter_api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing TAVILY_API_KEY or OPENROUTER_API_KEY in environment",
        )

    async def event_stream() -> AsyncIterator[dict]:
        original_query = body.query.strip()
        tracer = PipelineTracer(original_query)

        yield {"event": "status", "data": "Searching the web..."}
        with tracer.stage("retrieval_pipeline"):
            stage = await run_retrieval_stage(original_query)

        tracer.rewritten_query = stage.get("search_query")
        if stage.get("sub_queries"):
            tracer.record_cache_stats(
                {
                    **stage.get("fetch_stats", {}),
                    **stage.get("ingest_stats", {}),
                    "sub_queries": stage["sub_queries"],
                }
            )

        if stage["error"] == "No search results found.":
            yield {"event": "error", "data": stage["error"]}
            tracer.flush()
            return

        yield {"event": "sources", "data": json.dumps(stage["urls"])}

        fetch_stats = stage.get("fetch_stats", {})
        if fetch_stats.get("cache_hits"):
            yield {
                "event": "status",
                "data": (
                    f"L1 cache: {fetch_stats['cache_hits']} hit(s), "
                    f"{fetch_stats.get('fetched', 0)} fetched."
                ),
            }
        elif fetch_stats.get("failed"):
            yield {
                "event": "status",
                "data": (
                    f"Read {fetch_stats.get('fetched', 0)} of "
                    f"{len(stage['urls'])} sources successfully."
                ),
            }

        if stage["error"]:
            yield {"event": "error", "data": stage["error"]}
            tracer.flush()
            return

        top_chunks = stage["top_chunks"]
        tracer.record_chunks(top_chunks)

        yield {"event": "status", "data": "Generating answer..."}
        prompt = build_prompt(original_query, top_chunks)
        tracer.record_prompt_tokens(count_tokens(prompt))

        with tracer.stage("generation"):
            async for token in stream_llm_response(prompt):
                tracer.add_completion_tokens(count_tokens(token))
                yield {"event": "token", "data": token}

        if settings.enable_debug_events:
            yield {"event": "debug", "data": json.dumps(tracer.summary())}

        tracer.flush()
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())
