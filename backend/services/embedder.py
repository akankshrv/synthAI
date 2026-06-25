from sentence_transformers import CrossEncoder, SentenceTransformer

from core.config import settings

_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def _use_bge_query_prompt() -> bool:
    return "bge" in settings.embedding_model.lower()


def load_model() -> None:
    global _model, _cross_encoder
    _model = SentenceTransformer(settings.embedding_model)
    _cross_encoder = CrossEncoder(settings.cross_encoder_model)


def warm_models() -> None:
    embed_texts(["warmup"])
    embed_query("warmup")
    rerank("warmup", [{"text": "warmup passage for model init", "source": "warmup"}])


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _model is None:
        raise RuntimeError("Embedding model not loaded")
    if not texts:
        return []
    return _model.encode(texts, show_progress_bar=False, batch_size=64).tolist()


def embed_query(query: str) -> list[float]:
    if _model is None:
        raise RuntimeError("Embedding model not loaded")
    if _use_bge_query_prompt():
        vector = _model.encode([query], prompt_name="query", show_progress_bar=False)
    else:
        vector = _model.encode([query], show_progress_bar=False)
    return vector[0].tolist()


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    if _cross_encoder is None:
        raise RuntimeError("Cross-encoder model not loaded")
    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = _cross_encoder.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results: list[dict] = []
    for chunk, score in ranked:
        enriched = dict(chunk)
        enriched["rerank_score"] = round(float(score), 4)
        results.append(enriched)
    return results
