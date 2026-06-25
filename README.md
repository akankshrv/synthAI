# synthAI

Perplexity-style search with hybrid retrieval, cross-encoder reranking, and SSE streaming.

## Stack

- **Backend**: FastAPI, sentence-transformers, Chroma (ephemeral), BM25, SSE streaming
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
env -u PYTHONPATH uvicorn main:app --reload --port 8001
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000

If imports fail with mixed Python versions, recreate the venv without `PYTHONPATH`:

```bash
env -u PYTHONPATH python3.11 -m venv .venv
env -u PYTHONPATH .venv/bin/pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/uvicorn main:app --reload --port 8001
```

## Pipeline

```
POST /api/search
  → optional query rewrite (OpenRouter, feature-flagged)
  → Tavily (top 5 URLs, with retries)
  → Jina Reader (parallel fetch, per-URL retries)
  → token-aware chunking (RecursiveCharacterTextSplitter)
  → hybrid retrieval: dense (Chroma) + sparse (BM25) → RRF fusion
  → cross-encoder rerank (ms-marco-MiniLM-L-6-v2, top 8)
  → OpenRouter LLM stream → frontend SSE
  → optional debug trace event + traces.jsonl log
```

## Evaluation

```bash
cd backend
env -u PYTHONPATH .venv/bin/python eval/run_eval.py 3   # quick run (3 queries)
env -u PYTHONPATH .venv/bin/python eval/run_eval.py     # default 3 queries
env -u PYTHONPATH .venv/bin/python eval/test_sse_integration.py  # mocked SSE test
```

## Env vars

| Variable | Required | Description |
|----------|----------|-------------|
| `TAVILY_API_KEY` | yes | Web search |
| `OPENROUTER_API_KEY` | yes | LLM streaming |
| `JINA_API_KEY` | no | Higher Jina rate limits |
| `OPENROUTER_MODEL` | no | Default `openai/gpt-oss-120b:free` |
| `ENABLE_QUERY_REWRITE` | no | Default `true` |
| `QUERY_REWRITE_MODEL` | no | Default `openai/gpt-oss-20b:free` |
| `ENABLE_DEBUG_EVENTS` | no | Emit SSE `debug` events (default `true`) |
| `TRACE_LOG_PATH` | no | JSONL trace log (default `traces.jsonl`) |
| `TOP_K` | no | Final chunks sent to LLM (default `8`) |
| `RETRIEVAL_CANDIDATE_K` | no | Candidates per retriever before RRF (default `20`) |
