import asyncio
import logging
import re

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings

logger = logging.getLogger(__name__)

MAX_CHARS_PER_PAGE = 15_000

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(the\s+)?above\s+instructions",
        r"disregard\s+(all\s+)?prior\s+instructions",
        r"system\s+prompt",
        r"you\s+are\s+now",
        r"new\s+instructions\s*:",
        r"do\s+not\s+follow\s+(the\s+)?(above|previous)",
    ]
]

HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def sanitize_text(text: str) -> str:
    text = SCRIPT_STYLE_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def strip_injection_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = [line for line in lines if not contains_injection(line)]
    return "\n".join(cleaned).strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def _fetch_url(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    headers = {"Accept": "text/plain"}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"

    response = await client.get(
        f"https://r.jina.ai/{url}",
        headers=headers,
        timeout=settings.http_timeout,
    )
    response.raise_for_status()
    text = sanitize_text(response.text[:MAX_CHARS_PER_PAGE])
    return url, text


async def fetch_all_urls(urls: list[str]) -> tuple[dict[str, str], int]:
    pages: dict[str, str] = {}
    failed = 0

    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        results = await asyncio.gather(
            *[_fetch_url(client, url) for url in urls],
            return_exceptions=True,
        )

    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.warning("Failed to fetch %s after retries: %s", url, result)
            failed += 1
            pages[url] = ""
        else:
            _, text = result
            pages[url] = text

    succeeded = len(urls) - failed
    return pages, succeeded


def chunk_text(text: str, source: str) -> list[dict]:
    text = strip_injection_lines(sanitize_text(text))
    if not text:
        return []

    splitter = _get_splitter()
    raw_chunks = splitter.split_text(text)

    chunks: list[dict] = []
    for paragraph in raw_chunks:
        paragraph = paragraph.strip()
        if len(paragraph) < 80:
            continue
        if contains_injection(paragraph):
            logger.debug("Skipping chunk with injection pattern from %s", source)
            continue
        chunks.append({"text": paragraph, "source": source})
    return chunks
