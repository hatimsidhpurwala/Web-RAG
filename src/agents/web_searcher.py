"""
Web searcher agent – DuckDuckGo search, scrape-and-index, and
deep-research capabilities.

Key improvement: DuckDuckGo result snippets are NEVER discarded.
When a page blocks scraping or returns too little, the snippet
(title + URL + description) is stored as a lightweight fallback chunk.
This ensures dealer names, URLs, and descriptions always make it into
the knowledge base even if the full page is inaccessible.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from groq import Groq

from config.settings import (
    DEEP_RESEARCH_NUM_QUERIES,
    GROQ_API_KEY,
    LLM_MODEL,
    SCRAPE_DELAY_SECONDS,
    WEB_SEARCH_MAX_RESULTS,
)
from src.core.chunker import chunk_markdown
from src.core.cleaner import deduplicate_chunks, normalize_text
from src.core.embedder import embed_chunks
from src.core.scraper import scrape_website
from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ======================================================================
# Basic search — dual provider with automatic failover
# ======================================================================

def _search_duckduckgo(query: str, max_results: int) -> List[dict]:
    """Try DuckDuckGo first (fast, no API key needed).
    Automatically falls back across different backends if rate-limited.
    """
    backends = ["api", "lite", "html"]
    
    for backend in backends:
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results, backend=backend))
            if raw:
                logger.info("DuckDuckGo (%s) returned %d results for '%s'", backend, len(raw), query)
                return [
                    {
                        "title":   r.get("title", ""),
                        "url":     r.get("href", r.get("link", "")),
                        "snippet": r.get("body", ""),
                    }
                    for r in raw
                ]
        except Exception as exc:
            logger.warning("DuckDuckGo (%s) failed: %s", backend, exc)
            
    return []


def _search_google(query: str, max_results: int) -> List[dict]:
    """Fallback to Google via googlesearch-python."""
    try:
        from googlesearch import search as gsearch
        results: list[dict] = []
        for r in gsearch(query, num_results=max_results, advanced=True,
                         sleep_interval=2):
            results.append({
                "title":   getattr(r, "title", "") or "",
                "url":     getattr(r, "url", "") or "",
                "snippet": getattr(r, "description", "") or "",
            })
        logger.info("Google returned %d results for '%s'", len(results), query)
        return results
    except Exception as exc:
        logger.error("Google search also failed: %s", exc)
        return []


def search(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> List[dict]:
    """Search the web and return a list of {title, url, snippet} dicts.

    Uses DuckDuckGo as primary provider.  If DDG returns 0 results
    (usually due to rate-limiting), falls back to Google automatically.
    """
    results = _search_duckduckgo(query, max_results)
    if not results:
        logger.info("DDG returned 0 results — falling back to Google")
        results = _search_google(query, max_results)
    if not results:
        logger.warning("Both search providers returned 0 results for '%s'", query)
    return results


# ======================================================================
# Search + scrape + index
# ======================================================================

_MIN_SCRAPE_CHARS = 300   # if scrape returns less than this, use snippet


def search_and_scrape(
    query: str,
    vector_store: VectorStore,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> Dict:
    """Search the web, scrape top results, and index them.

    **Critical behaviour:** when a page cannot be scraped (blocked, JS-only,
    returns too little), the DuckDuckGo title + snippet is stored as a
    lightweight fallback chunk.  This ensures dealer names, URLs, and
    descriptions are never silently discarded.

    Returns
    -------
    dict
        ``sites_indexed`` (list of site labels),
        ``total_chunks``  (int),
        ``raw_results``   (list of {title, url, snippet} dicts — always populated).
    """
    results = search(query, max_results=max_results)
    sites_indexed: list[str] = []
    total_chunks = 0
    raw_results: list[dict] = []   # ← always returned for caller use

    for result in results:
        url = result.get("url", "")
        if not url:
            continue

        title   = result.get("title", "")
        snippet = result.get("snippet", "")

        # Build the rich snippet text that we use as fallback / supplement
        snippet_text = (
            f"**{title}**\n"
            f"Source: {url}\n\n"
            f"{snippet}"
        ).strip()

        # Save raw result for the caller (response generator can use this)
        raw_results.append({"title": title, "url": url, "snippet": snippet})

        # ── Try to scrape the full page ──────────────────────────────
        markdown: Optional[str] = None
        try:
            markdown = scrape_website(url, delay=SCRAPE_DELAY_SECONDS)
        except Exception as exc:
            logger.warning("Scrape error for %s: %s", url, exc)

        if not markdown or len(markdown.strip()) < _MIN_SCRAPE_CHARS:
            # Scraping failed or returned almost nothing — use snippet as content
            logger.info(
                "Using snippet fallback for %s (%d chars scraped)",
                url, len(markdown or "")
            )
            markdown = snippet_text

        markdown = normalize_text(markdown)
        domain    = urlparse(url).netloc.replace("www.", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        site_name = f"web_{domain}_{timestamp}"

        chunks = chunk_markdown(markdown, source_url=url)
        chunks = deduplicate_chunks(chunks)
        chunks = embed_chunks(chunks, show_progress=False)

        stored = vector_store.store_chunks_for_site(chunks, site_name)
        total_chunks += stored
        sites_indexed.append(site_name)

        logger.info("Indexed %d chunks from %s (via %s)",
                    stored, url, "scrape" if markdown != snippet_text else "snippet")

    return {
        "sites_indexed": sites_indexed,
        "total_chunks":  total_chunks,
        "raw_results":   raw_results,     # ← NEW: raw DDG results always included
    }


# ======================================================================
# Deep research
# ======================================================================

def _generate_research_queries(topic: str) -> List[str]:
    """Use the LLM to produce multiple targeted search queries for *topic*.

    Detects commercial intent (buy, dealer, distributor, where to find,
    supplier, reseller, shop) and generates queries with product model
    numbers and location terms for more specific results.
    """
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are a web-search query expert for a RAG system. "
        "Generate exactly {n} distinct search queries to find the best web pages "
        "for the given topic.\n\n"
        "RULES:\n"
        "1. Each query must be 4-8 words, no full sentences.\n"
        "2. Queries must be semantically diverse — no near-duplicates.\n"
        "3. Include product model numbers if mentioned (e.g. CCVD20xx, CCVD40xx).\n"
        "4. For COMMERCIAL queries (buy, dealer, distributor, where to find, "
        "supplier, reseller, price, shop, stock):\n"
        "   - Query 1: '[product model] buy online shop'\n"
        "   - Query 2: '[product] authorized distributor [location]'\n"
        "   - Query 3: '[product] reseller supplier dealer'\n"
        "   - Query 4: '[brand] official distributor [country]'\n"
        "   - Query 5: '[product model] price stock'\n"
        "5. For INFORMATIONAL queries: generate broad research queries.\n"
        "Return JSON: {\"queries\": [\"q1\", \"q2\", ...]}"
    ).format(n=DEEP_RESEARCH_NUM_QUERIES)

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic},
        ],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    queries = data.get("queries", [topic])
    logger.info("Deep research queries: %s", queries)
    return queries[:DEEP_RESEARCH_NUM_QUERIES]


def deep_research(
    topic: str,
    vector_store: VectorStore,
    num_queries: int = DEEP_RESEARCH_NUM_QUERIES,
) -> Dict:
    """Perform deep research: generate multiple queries, scrape, and index.

    Returns
    -------
    dict
        ``queries_used``, ``sites_indexed``, ``total_chunks``,
        ``raw_results`` (all DDG results collected across all queries).
    """
    queries = _generate_research_queries(topic)
    all_sites: list[str] = []
    all_raw: list[dict]  = []
    total_chunks = 0

    for query in queries:
        info = search_and_scrape(query, vector_store, max_results=2)
        all_sites.extend(info["sites_indexed"])
        all_raw.extend(info.get("raw_results", []))
        total_chunks += info["total_chunks"]

    # De-duplicate raw results by URL
    seen_urls: set[str] = set()
    deduped_raw: list[dict] = []
    for r in all_raw:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped_raw.append(r)

    logger.info(
        "Deep research complete: %d queries, %d sites, %d chunks, %d raw results",
        len(queries), len(all_sites), total_chunks, len(deduped_raw),
    )
    return {
        "queries_used":  queries,
        "sites_indexed": all_sites,
        "total_chunks":  total_chunks,
        "raw_results":   deduped_raw,   # ← NEW
    }
