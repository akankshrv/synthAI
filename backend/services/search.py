import httpx

from core.config import settings


async def tavily_search(query: str) -> list[str]:
    async with httpx.AsyncClient(timeout=30) as client:
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
