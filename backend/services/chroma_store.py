import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import chromadb

from core.config import settings
from services.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)

_client: Any = None
_collection: Any = None


def init_chroma() -> None:
    global _client, _collection
    path = Path(settings.chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(path))
    _collection = _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Chroma persistent store ready at %s", path)


def _collection_or_raise():
    if _collection is None:
        raise RuntimeError("Chroma store not initialized")
    return _collection


def _ttl_seconds() -> int:
    return int(settings.chroma_ttl_hours * 3600)


def _chunk_id(url: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{url}:{chunk_index}:{text}".encode()).hexdigest()
    return digest[:32]


def _url_chunk_metadata(url: str) -> dict | None:
    collection = _collection_or_raise()
    results = collection.get(where={"url": url}, include=["metadatas"], limit=1)
    if not results["ids"]:
        return None
    return results["metadatas"][0]


def _is_metadata_fresh(meta: dict) -> bool:
    fetched_at = meta.get("fetched_at")
    if fetched_at is None:
        return False
    return (time.time() - float(fetched_at)) <= _ttl_seconds()


def has_url_with_hash(url: str, page_hash: str) -> bool:
    meta = _url_chunk_metadata(url)
    if not meta:
        return False
    if meta.get("content_hash") != page_hash:
        return False
    if not _is_metadata_fresh(meta):
        logger.info("Chroma TTL expired for %s — will re-ingest", url)
        delete_chunks_for_url(url)
        return False
    return True


def delete_chunks_for_url(url: str) -> None:
    collection = _collection_or_raise()
    existing = collection.get(where={"url": url}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.debug("Deleted %d chunks for %s", len(existing["ids"]), url)


def upsert_url_chunks(
    url: str,
    page_hash: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    collection = _collection_or_raise()
    fetched_at = time.time()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        ids.append(_chunk_id(url, i, chunk["text"]))
        documents.append(chunk["text"])
        metadatas.append(
            {
                "url": url,
                "source": url,
                "content_hash": page_hash,
                "chunk_index": i,
                "fetched_at": fetched_at,
            }
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def get_chunks_for_urls(urls: list[str]) -> list[dict]:
    if not urls:
        return []

    collection = _collection_or_raise()
    results = collection.get(
        where={"url": {"$in": urls}},
        include=["documents", "metadatas"],
    )

    chunks: list[dict] = []
    for chunk_id, doc, meta in zip(
        results["ids"], results["documents"], results["metadatas"]
    ):
        if not _is_metadata_fresh(meta):
            continue
        chunks.append(
            {
                "id": chunk_id,
                "text": doc,
                "source": meta.get("source", meta.get("url", "")),
                "url": meta.get("url", ""),
                "content_hash": meta.get("content_hash", ""),
            }
        )
    return chunks


def query_dense(
    query: str,
    urls: list[str],
    candidate_k: int,
) -> tuple[list[str], dict[str, float], dict[str, dict]]:
    if not urls:
        return [], {}, {}

    collection = _collection_or_raise()
    fresh_chunks = get_chunks_for_urls(urls)
    if not fresh_chunks:
        return [], {}, {}

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, len(fresh_chunks)),
        where={"url": {"$in": urls}},
        include=["documents", "metadatas", "distances"],
    )

    ranked_ids: list[str] = []
    dense_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    fresh_ids = {chunk["id"] for chunk in fresh_chunks}

    for chunk_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if chunk_id not in fresh_ids:
            continue
        ranked_ids.append(chunk_id)
        dense_scores[chunk_id] = round(1.0 / (1.0 + float(distance)), 4)
        chunk_map[chunk_id] = {
            "id": chunk_id,
            "text": doc,
            "source": meta.get("source", meta.get("url", "")),
            "url": meta.get("url", ""),
        }

    return ranked_ids, dense_scores, chunk_map


def purge_expired_chunks() -> int:
    """Delete orphan chunks past TTL. Returns number of chunks removed."""
    collection = _collection_or_raise()
    results = collection.get(include=["metadatas"])
    expired_ids = [
        chunk_id
        for chunk_id, meta in zip(results["ids"], results["metadatas"])
        if not _is_metadata_fresh(meta)
    ]
    if expired_ids:
        collection.delete(ids=expired_ids)
        logger.info("Purged %d expired Chroma chunks", len(expired_ids))
    return len(expired_ids)
