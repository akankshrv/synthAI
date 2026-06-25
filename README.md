# synthAI

Perplexity-style search with hybrid retrieval, cross-encoder reranking, and SSE streaming.

## Stack

- **Backend**: FastAPI, BGE embeddings, persistent Chroma, Redis, BM25, SSE streaming
- **Frontend**: Next.js

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add TAVILY_API_KEY and OPENROUTER_API_KEY
docker compose up -d redis   # or local redis-server
env -u PYTHONPATH uvicorn main:app --reload --port 8001
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Pipeline

```
POST /api/search
  → query rewrite + decomposition (1-3 sub-queries)
  → Tavily per sub-query (up to 8 URLs)
  → L1 Redis page cache (TTL 24h)
  → Jina for cache misses
  → token-aware chunking (500 tokens, 50 overlap)
  → Chroma ingest (hash + TTL 72h)
  → dense (BGE) + BM25 → RRF → dedup → rerank → MMR → top 8
  → OpenRouter LLM stream + debug SSE
```

## Evaluation

```bash
cd backend
env -u PYTHONPATH .venv/bin/python eval/run_eval.py 3 --save-baseline
env -u PYTHONPATH .venv/bin/python eval/run_eval.py 3 --no-rerank --no-bm25
env -u PYTHONPATH .venv/bin/python eval/test_sse_integration.py
```

Ablation flags: `--no-rerank`, `--no-bm25`, `--no-cache`, `--no-mmr`, `--no-dedup`, `--no-decompose`

Baselines saved to `eval/baselines/`.

## Key env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense retrieval model |
| `CHUNK_SIZE_TOKENS` | `500` | Token-aware chunk size |
| `REDIS_URL` | `redis://localhost:6379/0` | L1 page cache |
| `CHROMA_TTL_HOURS` | `72` | Chroma chunk TTL |
| `ENABLE_QUERY_DECOMPOSITION` | `true` | Multi sub-query search |
