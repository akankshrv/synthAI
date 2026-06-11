import json
from collections.abc import AsyncIterator

import httpx

from core.config import settings

SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question using ONLY the provided sources.
Cite sources inline as [1], [2], etc. matching the source numbers given.
If the sources do not contain enough information, say so clearly. Do not invent facts."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {chunk['text']}\nSource: {chunk['source']}"
        for i, chunk in enumerate(chunks)
    )
    return f"Sources:\n{context}\n\nQuestion: {query}"


async def stream_llm_response(prompt: str) -> AsyncIterator[str]:
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "synthAI",
    }
    payload = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
