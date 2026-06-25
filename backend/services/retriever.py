import re

from rank_bm25 import BM25Okapi

from core.config import settings
from services import chroma_store
from services.chunk_utils import apply_mmr, dedup_chunks
from services.embedder import rerank

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    rrf_k = k or settings.rrf_k
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _bm25_search(
    query: str,
    chunks: list[dict],
    candidate_k: int,
) -> tuple[list[str], dict[str, float]]:
    if not chunks:
        return [], {}

    corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    candidate_ids = [
        chunks[i]["id"] for i in ranked_indices[: min(candidate_k, len(chunks))]
    ]
    bm25_scores = {
        chunks[i]["id"]: round(float(scores[i]), 4)
        for i in ranked_indices[: min(candidate_k, len(chunks))]
    }
    return candidate_ids, bm25_scores


def retrieve_top_chunks(queries: str | list[str], urls: list[str]) -> list[dict]:
    """Retrieve top chunks for one or more queries scoped to source URLs."""
    if not urls:
        return []

    query_list = [queries] if isinstance(queries, str) else [q for q in queries if q]
    if not query_list:
        return []

    all_chunks = chroma_store.get_chunks_for_urls(urls)
    if not all_chunks:
        return []

    chunk_map = {chunk["id"]: chunk for chunk in all_chunks}
    candidate_k = min(settings.retrieval_candidate_k, len(all_chunks))
    primary_query = query_list[0]

    rankings: list[list[str]] = []
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}

    for query in query_list:
        dense_ids, query_dense_scores, dense_map = chroma_store.query_dense(
            query, urls, candidate_k
        )
        chunk_map.update(dense_map)
        rankings.append(dense_ids)
        dense_scores.update(query_dense_scores)

        if settings.enable_bm25:
            bm25_ids, query_bm25_scores = _bm25_search(query, all_chunks, candidate_k)
            rankings.append(bm25_ids)
            bm25_scores.update(query_bm25_scores)

    fused = reciprocal_rank_fusion(rankings)
    fused = fused[: min(settings.rrf_top_n, len(fused))]

    fused_candidates: list[dict] = []
    for chunk_id, rrf_score in fused:
        chunk = dict(chunk_map[chunk_id])
        chunk["id"] = chunk_id
        chunk["dense_score"] = dense_scores.get(chunk_id)
        chunk["bm25_score"] = bm25_scores.get(chunk_id)
        chunk["rrf_score"] = round(rrf_score, 4)
        fused_candidates.append(chunk)

    fused_candidates = dedup_chunks(fused_candidates)

    if settings.enable_rerank:
        scored = rerank(primary_query, fused_candidates)
    else:
        scored = fused_candidates

    final = apply_mmr(primary_query, scored, settings.top_k)

    for i, chunk in enumerate(final, start=1):
        chunk["citation_id"] = i

    return final
