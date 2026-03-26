"""
utils/web_research.py

Free web research for company data.
Uses DuckDuckGo (no API key) for search and requests/BeautifulSoup for scraping.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_TIMEOUT = 8          # seconds per HTTP request
_MAX_PAGE_CHARS = 2000  # chars to keep per scraped page

# Domains that are unhelpful to scrape
_SKIP_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "tiktok.com", "reddit.com",
}


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    """Run a DuckDuckGo text search. Returns list of result dicts."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
        return []


def _scrape_url(url: str) -> str:
    """Fetch a URL and return stripped plain text, capped at _MAX_PAGE_CHARS."""
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text)
        return text[:_MAX_PAGE_CHARS]
    except Exception as exc:
        logger.debug("Scrape failed for %s: %s", url, exc)
        return ""


def research_company(company_name: str, domain: str) -> dict:
    """
    Search the web and scrape pages to gather structured company intelligence.

    Returns a dict with:
        search_snippets  — list of {title, snippet, url} from general search
        scraped_pages    — list of {url, text} from the top crawlable pages
        news_snippets    — list of {title, snippet} from a news-focused search
    """
    if not company_name or company_name.startswith("N/A"):
        logger.debug("Skipping web research for non-corporate contact.")
        return {"search_snippets": [], "scraped_pages": [], "news_snippets": []}

    # ── 1. General company search ─────────────────────────────────────────────
    general_query = (
        f"{company_name} company overview revenue employees headquarters founded"
    )
    general_results = _ddg_search(general_query, max_results=6)

    search_snippets = [
        {
            "title":   r.get("title", ""),
            "snippet": r.get("body", ""),
            "url":     r.get("href", ""),
        }
        for r in general_results
    ]

    # ── 2. Scrape top 2 crawlable pages ───────────────────────────────────────
    scraped_pages: list[dict] = []
    for result in general_results:
        url = result.get("href", "")
        if not url:
            continue
        if any(skip in url for skip in _SKIP_DOMAINS):
            continue
        if url.lower().endswith(".pdf"):
            continue
        text = _scrape_url(url)
        if text:
            scraped_pages.append({"url": url, "text": text})
        if len(scraped_pages) >= 2:
            break

    # ── 3. Recent news search ─────────────────────────────────────────────────
    news_query = f"{company_name} news 2025"
    news_results = _ddg_search(news_query, max_results=5)
    news_snippets = [
        {"title": r.get("title", ""), "snippet": r.get("body", "")}
        for r in news_results
    ]

    logger.info(
        "Web research for %r: %d snippets, %d pages scraped, %d news items",
        company_name,
        len(search_snippets),
        len(scraped_pages),
        len(news_snippets),
    )

    return {
        "search_snippets": search_snippets,
        "scraped_pages":   scraped_pages,
        "news_snippets":   news_snippets,
    }


def format_web_data_for_prompt(web_data: dict) -> str:
    """
    Convert raw web research data into a compact, LLM-readable block.
    Keeps total size manageable so it fits inside an LLM context window.
    """
    if not web_data:
        return ""

    lines: list[str] = []

    snippets = web_data.get("search_snippets", [])
    if snippets:
        lines.append("Web search results:")
        for i, s in enumerate(snippets[:5], 1):
            title   = s.get("title", "")
            snippet = (s.get("snippet") or "")[:220]
            lines.append(f"  {i}. {title}: {snippet}")

    scraped = web_data.get("scraped_pages", [])
    if scraped:
        lines.append("\nScraped page content:")
        for page in scraped[:2]:
            url  = page.get("url", "")
            text = (page.get("text") or "")[:1500]
            lines.append(f"  [{url}]\n  {text}")

    news = web_data.get("news_snippets", [])
    if news:
        lines.append("\nRecent news headlines:")
        for i, n in enumerate(news[:5], 1):
            title   = n.get("title", "")
            snippet = (n.get("snippet") or "")[:160]
            lines.append(f"  {i}. {title}: {snippet}")

    return "\n".join(lines)
