import hashlib
import json
import logging
import time
from typing import Any

import redis

from core.config import settings

logger = logging.getLogger(__name__)

_client: Any = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(url: str) -> str:
    url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{settings.redis_key_prefix}:page:{url_digest}"


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def init_redis() -> None:
    client = _get_client()
    client.ping()
    logger.info("Redis page cache ready at %s", settings.redis_url)


def get(url: str) -> tuple[str, str, float] | None:
    """Return (content, content_hash, fetched_at) if present, else None."""
    raw = _get_client().get(_cache_key(url))
    if not raw:
        return None

    payload = json.loads(raw)
    return payload["content"], payload["content_hash"], float(payload["fetched_at"])


def set(url: str, content: str) -> str:
    page_hash = content_hash(content)
    ttl_seconds = int(settings.url_cache_ttl_hours * 3600)
    payload = json.dumps(
        {
            "content": content,
            "content_hash": page_hash,
            "fetched_at": time.time(),
        }
    )
    _get_client().set(_cache_key(url), payload, ex=ttl_seconds)
    return page_hash


def partition_urls(urls: list[str]) -> tuple[dict[str, str], list[str]]:
    """Split URLs into L1 cache hits and misses."""
    hits: dict[str, str] = {}
    misses: list[str] = []
    for url in urls:
        cached = get(url)
        if cached:
            hits[url] = cached[0]
        else:
            misses.append(url)
    return hits, misses
