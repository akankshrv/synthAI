import chromadb

from services.embedder import embed_texts

TOP_K = 8


def retrieve_top_chunks(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

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
        n_results=min(TOP_K, len(chunks)),
    )

    top_chunks: list[dict] = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        top_chunks.append({"text": doc, "source": metadata["source"]})
    return top_chunks
