import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from models.schemas import SearchRequest
from services.llm import build_prompt, count_tokens, rewrite_query, stream_llm_response
from services.retriever import retrieve_top_chunks
from services.scraper import chunk_text, fetch_all_urls
from services.search import tavily_search
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

        with tracer.stage("query_rewrite"):
            search_query = await rewrite_query(original_query)
            tracer.rewritten_query = search_query

        yield {"event": "status", "data": "Searching the web..."}
        try:
            with tracer.stage("tavily_search"):
                urls = await tavily_search(search_query)
        except Exception:
            yield {"event": "error", "data": "Web search failed. Please try again."}
            tracer.flush()
            return

        if not urls:
            yield {"event": "error", "data": "No search results found."}
            tracer.flush()
            return

        yield {"event": "sources", "data": json.dumps(urls)}

        yield {"event": "status", "data": "Reading pages..."}
        with tracer.stage("fetch_pages"):
            pages, succeeded = await fetch_all_urls(urls)

        if succeeded < len(urls):
            yield {
                "event": "status",
                "data": f"Read {succeeded} of {len(urls)} sources successfully.",
            }

        chunks: list[dict] = []
        with tracer.stage("chunking"):
            for url, text in pages.items():
                if text:
                    chunks.extend(chunk_text(text, url))

        if not chunks:
            yield {
                "event": "error",
                "data": "Could not extract content from search results.",
            }
            tracer.flush()
            return

        yield {"event": "status", "data": "Finding relevant passages..."}
        with tracer.stage("retrieval"):
            top_chunks = retrieve_top_chunks(search_query, chunks)
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
