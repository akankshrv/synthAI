import logging

from core.config import settings
from services import chroma_store, url_cache
from services.embedder import embed_texts
from services.scraper import chunk_text, fetch_all_urls

logger = logging.getLogger(__name__)


async def fetch_pages(urls: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    """
    Resolve page text for URLs using L1 Redis cache, fetching only misses.

    Returns (pages, stats) where stats has cache_hits, fetched, failed.
    """
    if not settings.enable_cache:
        pages, succeeded = await fetch_all_urls(urls)
        return (
            {url: pages.get(url, "") for url in urls},
            {"cache_hits": 0, "fetched": succeeded, "failed": len(urls) - succeeded},
        )

    hits, misses = url_cache.partition_urls(urls)
    stats = {"cache_hits": len(hits), "fetched": 0, "failed": 0}

    pages = dict(hits)
    if misses:
        fetched, succeeded = await fetch_all_urls(misses)
        stats["fetched"] = succeeded
        stats["failed"] = len(misses) - succeeded
        for url, text in fetched.items():
            pages[url] = text
            if text:
                url_cache.set(url, text)

    return {url: pages.get(url, "") for url in urls}, stats


def ingest_pages(urls: list[str], pages: dict[str, str]) -> dict[str, int]:
    """
    Chunk, embed, and upsert into persistent Chroma for new or changed pages.

    Skips embed when Chroma already has matching content_hash for the URL
    and fetched_at is within CHROMA_TTL_HOURS.
    """
    stats = {"chroma_skipped": 0, "chroma_ingested": 0, "empty": 0}

    for url in urls:
        text = pages.get(url, "")
        if not text:
            stats["empty"] += 1
            continue

        page_hash = url_cache.content_hash(text)
        if (
            settings.enable_cache
            and chroma_store.has_url_with_hash(url, page_hash)
        ):
            stats["chroma_skipped"] += 1
            continue

        chroma_store.delete_chunks_for_url(url)
        chunks = chunk_text(text, url)
        if not chunks:
            stats["empty"] += 1
            continue

        embeddings = embed_texts([chunk["text"] for chunk in chunks])
        chroma_store.upsert_url_chunks(url, page_hash, chunks, embeddings)
        stats["chroma_ingested"] += 1
        logger.info("Ingested %d chunks for %s", len(chunks), url)

    return stats
