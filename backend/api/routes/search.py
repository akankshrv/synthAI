import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from models.schemas import SearchRequest
from services.llm import build_prompt, stream_llm_response
from services.retriever import retrieve_top_chunks
from services.scraper import chunk_text, fetch_all_urls
from services.search import tavily_search

router = APIRouter()


@router.post("/search")
async def search(body: SearchRequest) -> EventSourceResponse:
    if not settings.tavily_api_key or not settings.openrouter_api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing TAVILY_API_KEY or OPENROUTER_API_KEY in environment",
        )

    async def event_stream() -> AsyncIterator[dict]:
        query = body.query.strip()

        yield {"event": "status", "data": "Searching the web..."}
        urls = await tavily_search(query)
        if not urls:
            yield {"event": "error", "data": "No search results found."}
            return

        yield {"event": "sources", "data": json.dumps(urls)}

        yield {"event": "status", "data": "Reading pages..."}
        pages = await fetch_all_urls(urls)

        chunks: list[dict] = []
        for url, text in pages.items():
            if text:
                chunks.extend(chunk_text(text, url))

        if not chunks:
            yield {"event": "error", "data": "Could not extract content from search results."}
            return

        yield {"event": "status", "data": "Finding relevant passages..."}
        top_chunks = retrieve_top_chunks(query, chunks)

        yield {"event": "status", "data": "Generating answer..."}
        prompt = build_prompt(query, top_chunks)
        async for token in stream_llm_response(prompt):
            yield {"event": "token", "data": token}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())
