import re

import chromadb
from rank_bm25 import BM25Okapi

from core.config import settings
from services.embedder import embed_texts, rerank

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int | None = None,
) -> list[tuple[int, float]]:
    """Merge ranked chunk-id lists using reciprocal rank fusion."""
    rrf_k = k or settings.rrf_k
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _dense_search(
    query: str,
    chunks: list[dict],
    candidate_k: int,
) -> tuple[list[int], dict[int, float]]:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    query_embedding = embed_texts([query])[0]

    client = chromadb.EphemeralClient()
    collection = client.create_collection("query")

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, len(chunks)),
    )

    ranked_ids: list[int] = []
    dense_scores: dict[int, float] = {}
    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        chunk_id = int(doc_id)
        ranked_ids.append(chunk_id)
        dense_scores[chunk_id] = round(1.0 / (1.0 + float(distance)), 4)

    return ranked_ids, dense_scores


def _bm25_search(
    query: str,
    chunks: list[dict],
    candidate_k: int,
) -> tuple[list[int], dict[int, float]]:
    corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    candidate_ids = ranked[: min(candidate_k, len(chunks))]
    bm25_scores = {i: round(float(scores[i]), 4) for i in candidate_ids}
    return candidate_ids, bm25_scores


def retrieve_top_chunks(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    candidate_k = min(settings.retrieval_candidate_k, len(chunks))

    dense_ids, dense_scores = _dense_search(query, chunks, candidate_k)
    bm25_ids, bm25_scores = _bm25_search(query, chunks, candidate_k)

    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])
    fused_candidates: list[dict] = []
    for chunk_id, rrf_score in fused:
        chunk = dict(chunks[chunk_id])
        chunk["id"] = chunk_id
        chunk["dense_score"] = dense_scores.get(chunk_id)
        chunk["bm25_score"] = bm25_scores.get(chunk_id)
        chunk["rrf_score"] = round(rrf_score, 4)
        fused_candidates.append(chunk)

    reranked = rerank(query, fused_candidates)

    for i, chunk in enumerate(reranked, start=1):
        chunk["citation_id"] = i

    return reranked
