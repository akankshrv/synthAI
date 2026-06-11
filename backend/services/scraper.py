import asyncio

import httpx

from core.config import settings

MAX_CHARS_PER_PAGE = 15_000


async def _fetch_url(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    headers = {"Accept": "text/plain"}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"

    try:
        response = await client.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        text = response.text[:MAX_CHARS_PER_PAGE]
        return url, text
    except Exception:
        return url, ""


async def fetch_all_urls(urls: list[str]) -> dict[str, str]:
    async with httpx.AsyncClient() as client:
        pairs = await asyncio.gather(*[_fetch_url(client, url) for url in urls])
    return dict(pairs)


def chunk_text(text: str, source: str) -> list[dict]:
    chunks: list[dict] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if len(paragraph) >= 80:
            chunks.append({"text": paragraph, "source": source})
    return chunks
