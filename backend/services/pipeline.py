"""Shared RAG pipeline orchestration for API and evaluation."""

from core.config import settings
from services import chroma_store
from services.chroma_store import init_chroma
from services.embedder import load_model
from services.ingest import fetch_pages, ingest_pages
from services.llm import (
    build_prompt,
    decompose_query,
    rewrite_query,
    stream_llm_response,
)
from services.retriever import retrieve_top_chunks
from services.search import tavily_search


async def _gather_search_urls(search_queries: list[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for search_query in search_queries:
        found = await tavily_search(search_query)
        for url in found:
            if url not in seen:
                seen.add(url)
                urls.append(url)
        if len(urls) >= settings.max_search_urls:
            break

    return urls[: settings.max_search_urls]


async def run_retrieval_stage(original_query: str) -> dict:
    """Run search → fetch → ingest → retrieve. Returns stage outputs or error."""
    search_query = await rewrite_query(original_query)
    sub_queries = await decompose_query(search_query)

    urls = await _gather_search_urls(sub_queries)
    if not urls:
        return {
            "error": "No search results found.",
            "original_query": original_query,
            "search_query": search_query,
            "sub_queries": sub_queries,
            "urls": [],
            "fetch_stats": {},
            "ingest_stats": {},
            "top_chunks": [],
        }

    pages, fetch_stats = await fetch_pages(urls)
    ingest_stats = ingest_pages(urls, pages)

    usable_urls = [url for url in urls if pages.get(url)]
    if not usable_urls or not chroma_store.get_chunks_for_urls(usable_urls):
        return {
            "error": "Could not extract content from search results.",
            "original_query": original_query,
            "search_query": search_query,
            "sub_queries": sub_queries,
            "urls": urls,
            "fetch_stats": fetch_stats,
            "ingest_stats": ingest_stats,
            "top_chunks": [],
        }

    top_chunks = retrieve_top_chunks(sub_queries, usable_urls)
    return {
        "error": None,
        "original_query": original_query,
        "search_query": search_query,
        "sub_queries": sub_queries,
        "urls": urls,
        "fetch_stats": fetch_stats,
        "ingest_stats": ingest_stats,
        "top_chunks": top_chunks,
    }


async def run_pipeline(query: str) -> dict:
    """Run the full RAG pipeline and return structured results."""
    init_chroma()
    load_model()

    stage = await run_retrieval_stage(query.strip())
    if stage["error"]:
        return {
            "original_query": stage["original_query"],
            "search_query": stage["search_query"],
            "sub_queries": stage.get("sub_queries", []),
            "urls": stage["urls"],
            "contexts": [],
            "answer": "",
            "error": stage["error"],
        }

    top_chunks = stage["top_chunks"]
    prompt = build_prompt(stage["original_query"], top_chunks)
    answer_parts: list[str] = []
    async for token in stream_llm_response(prompt):
        answer_parts.append(token)

    return {
        "original_query": stage["original_query"],
        "search_query": stage["search_query"],
        "sub_queries": stage["sub_queries"],
        "urls": stage["urls"],
        "contexts": [c["text"] for c in top_chunks],
        "answer": "".join(answer_parts),
        "error": None,
    }
