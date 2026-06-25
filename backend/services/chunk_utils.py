"""Chunk deduplication and MMR diversity selection."""

import numpy as np

from core.config import settings
from services.embedder import embed_texts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def dedup_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks or not settings.enable_dedup:
        return chunks

    embeddings = _normalize(np.array(embed_texts([c["text"] for c in chunks])))
    kept: list[dict] = []
    kept_embeddings: list[np.ndarray] = []
    threshold = settings.dedup_similarity_threshold

    for idx, chunk in enumerate(chunks):
        if not kept:
            kept.append(chunk)
            kept_embeddings.append(embeddings[idx])
            continue
        sims = [float(np.dot(embeddings[idx], prev)) for prev in kept_embeddings]
        if max(sims) < threshold:
            kept.append(chunk)
            kept_embeddings.append(embeddings[idx])

    return kept


def apply_mmr(query: str, chunks: list[dict], k: int | None = None) -> list[dict]:
    if not chunks or not settings.enable_mmr:
        return chunks[: k or settings.top_k]

    limit = k or settings.top_k
    texts = [chunk["text"] for chunk in chunks]
    vectors = _normalize(np.array(embed_texts([query, *texts])))
    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    query_sims = doc_vecs @ query_vec

    selected: list[dict] = []
    selected_idx: list[int] = []
    remaining = list(range(len(chunks)))
    lambda_mult = settings.mmr_lambda

    while remaining and len(selected) < limit:
        best_score = float("-inf")
        best_idx = remaining[0]
        for idx in remaining:
            relevance = float(query_sims[idx])
            if not selected_idx:
                score = relevance
            else:
                redundancy = max(float(doc_vecs[idx] @ doc_vecs[prev]) for prev in selected_idx)
                score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(chunks[best_idx])
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return selected
