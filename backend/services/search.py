import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def tavily_search(query: str) -> list[str]:
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": 5,
                "include_raw_content": False,
            },
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    return [item["url"] for item in results if item.get("url")]
