from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    _model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _model is None:
        raise RuntimeError("Embedding model not loaded")
    return _model.encode(texts, show_progress_bar=False).tolist()
