"""
src/agent/web_search.py
DuckDuckGo search + fetch URL + embed result
"""

import logging

import html2text
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from src.ingestion.chunker import process
from src.storage.vector_store import embed_chunks

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5) -> list[str]:
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                if "href" in r:
                    urls.append(r["href"])
    except Exception as e:
        logger.error(f"Search failed: {e}")
    return urls


def fetch_and_embed(url: str, session_id: str, vector_store) -> None:
    site_name = f"{session_id}:web:{url}"
    if vector_store.has_site(site_name):
        return

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in ["script", "style", "nav", "footer", "aside"]:
            for el in soup.find_all(tag):
                el.decompose()

        h2t = html2text.HTML2Text()
        h2t.ignore_links = True
        h2t.ignore_images = True
        md_text = h2t.handle(str(soup))

        chunks_str = process(md_text)
        if not chunks_str:
            return

        chunks = [{"text": c, "source_url": url} for c in chunks_str]
        chunks = embed_chunks(chunks)
        vector_store.store_chunks_for_site(chunks, site_name)
    except Exception as e:
        logger.error(f"Fetch failed for {url}: {e}")
