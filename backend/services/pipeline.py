"""Shared pipeline runner for API and evaluation."""

from services.llm import build_prompt, rewrite_query, stream_llm_response
from services.retriever import retrieve_top_chunks
from services.scraper import chunk_text, fetch_all_urls
from services.search import tavily_search


async def run_pipeline(query: str) -> dict:
    """Run the full RAG pipeline and return structured results."""
    original_query = query.strip()
    search_query = await rewrite_query(original_query)

    urls = await tavily_search(search_query)
    if not urls:
        return {
            "original_query": original_query,
            "search_query": search_query,
            "urls": [],
            "contexts": [],
            "answer": "",
            "error": "No search results found.",
        }

    pages, _ = await fetch_all_urls(urls)
    chunks: list[dict] = []
    for url, text in pages.items():
        if text:
            chunks.extend(chunk_text(text, url))

    if not chunks:
        return {
            "original_query": original_query,
            "search_query": search_query,
            "urls": urls,
            "contexts": [],
            "answer": "",
            "error": "Could not extract content from search results.",
        }

    top_chunks = retrieve_top_chunks(search_query, chunks)
    contexts = [c["text"] for c in top_chunks]

    prompt = build_prompt(original_query, top_chunks)
    answer_parts: list[str] = []
    async for token in stream_llm_response(prompt):
        answer_parts.append(token)

    return {
        "original_query": original_query,
        "search_query": search_query,
        "urls": urls,
        "contexts": contexts,
        "answer": "".join(answer_parts),
        "error": None,
    }
