import asyncio
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
        history = [turn.model_dump() for turn in body.history]
        prior_urls = body.prior_urls[: settings.max_search_urls]
        tracer = PipelineTracer(original_query)
        stage_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_stage(stage_id: str, state: str, label: str | None = None) -> None:
            await stage_queue.put(
                {
                    "event": "stage",
                    "data": json.dumps(
                        {"id": stage_id, "state": state, "label": label or ""}
                    ),
                }
            )

        retrieval_task = asyncio.create_task(
            run_retrieval_stage(
                original_query,
                on_stage=on_stage,
                history=history,
                prior_urls=prior_urls,
            )
        )

        while not retrieval_task.done() or not stage_queue.empty():
            try:
                yield stage_queue.get_nowait()
            except asyncio.QueueEmpty:
                if retrieval_task.done():
                    break
                await asyncio.sleep(0.05)

        stage = await retrieval_task

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

        yield {
            "event": "stage",
            "data": json.dumps(
                {
                    "id": "generate",
                    "state": "active",
                    "label": "Generating answer",
                }
            ),
        }

        prompt = build_prompt(original_query, top_chunks, history=history)
        tracer.record_prompt_tokens(count_tokens(prompt))

        with tracer.stage("generation"):
            async for token in stream_llm_response(
                original_query,
                top_chunks,
                history=history,
            ):
                tracer.add_completion_tokens(count_tokens(token))
                yield {"event": "token", "data": token}

        yield {
            "event": "stage",
            "data": json.dumps(
                {"id": "generate", "state": "done", "label": "Generating answer"}
            ),
        }

        if settings.enable_debug_events:
            yield {"event": "debug", "data": json.dumps(tracer.summary())}

        tracer.flush()
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())
