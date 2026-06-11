# synthAI

Perplexity-style search: Tavily → Jina Reader → Chroma retrieval → OpenRouter streaming.

## Stack

- **Backend**: FastAPI, sentence-transformers, Chroma (ephemeral), SSE streaming
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
uvicorn main:app --reload --port 8001
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
  → Tavily (top 5 URLs)
  → Jina Reader (parallel fetch)
  → paragraph chunking
  → sentence-transformers embeddings
  → Chroma EphemeralClient (top 8 chunks)
  → OpenRouter LLM stream → frontend SSE
```

## Env vars

| Variable | Required | Description |
|----------|----------|-------------|
| `TAVILY_API_KEY` | yes | Web search |
| `OPENROUTER_API_KEY` | yes | LLM streaming |
| `JINA_API_KEY` | no | Higher Jina rate limits |
| `OPENROUTER_MODEL` | no | Default `openai/gpt-oss-120b:free` |
