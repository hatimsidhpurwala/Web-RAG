"""
Retriever agent – takes generated queries, embeds them, searches
Qdrant, deduplicates, and returns the top-k most relevant chunks.

Key improvement: accepts an optional ``source_prefix`` that restricts
the search to chunks from a specific source type (e.g. ``"pdf_"`` for
uploaded documents).  When a prefix is given but returns no results,
it automatically falls back to searching all sources so the user always
gets an answer.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.core.embedder import embed_query
from src.database.vector_store import VectorStore
from config.settings import TOP_K_PER_QUERY, TOP_K_RESULTS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def retrieve_chunks(
    queries: List[str],
    vector_store: VectorStore,
    *,
    top_k_per_query: int = TOP_K_PER_QUERY,
    final_top_k: int = TOP_K_RESULTS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    source_prefix: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[dict]:
    """Retrieve and deduplicate the most relevant chunks for *queries*.

    Parameters
    ----------
    queries : list[str]
        Optimised search queries from the query generator.
    vector_store : VectorStore
        The Qdrant wrapper instance.
    top_k_per_query : int
        Number of results to fetch per query.
    final_top_k : int
        Maximum number of chunks to return after deduplication.
    similarity_threshold : float
        Minimum score to accept a chunk.
    source_prefix : str | None
        If given (e.g. ``"pdf_"``), restrict the search to chunks whose
        ``site_name`` starts with this prefix.  Falls back to all sources
        if no results are found with the prefix filter.
    session_id : str | None
        When provided, all searches are further constrained to only chunks
        whose ``site_name`` begins with ``"{session_id}:"`` or the PDF
        prefix ``"pdf_{session_id}_"`` so different users' data never mix.
        When *source_prefix* is also given it takes precedence.

    Returns
    -------
    list[dict]
        Sorted by descending relevance score.
    """
    all_results: list[dict] = []
    seen_texts: set[str] = set()
    effective_top_k = max(top_k_per_query, 5)

    # When session_id is given and no explicit source_prefix, scope the search
    # to the current user's namespace so other users' chunks are invisible.
    # The namespace prefix for session-scoped web chunks is "{session_id}:"
    # and for PDF chunks it is "pdf_{session_id}_" (set by streamlit_app.py).
    # If source_prefix is already set (e.g. "pdf_abc123_") it takes precedence.
    if session_id and not source_prefix:
        source_prefix = f"{session_id}:"
        logger.debug(
            "Retriever: auto-scoping search to session namespace '%s'", source_prefix
        )

    def _search(query: str, prefix: Optional[str], threshold: Optional[float]) -> List[dict]:
        """Embed query and search – with or without source prefix.

        When a source prefix is given we disable the score threshold:
        generic "what is this about" questions score very low (≈ 0.18)
        against specific technical docs, so threshold filtering would
        silently drop all results even though the chunks ARE correct.
        """
        query_vector = embed_query(query)
        if prefix:
            # Always retrieve ALL chunks from this source, ignoring threshold
            return vector_store.search_chunks_by_prefix(
                query_vector=query_vector,
                site_prefix=prefix,
                top_k=effective_top_k,
                score_threshold=None,   # ← disabled intentionally
            )
        return vector_store.search_chunks(
            query_vector=query_vector,
            top_k=effective_top_k,
            score_threshold=threshold,
        )


    def _collect(chunks: List[dict]) -> None:
        for chunk in chunks:
            key = chunk["text"][:200]
            if key not in seen_texts:
                seen_texts.add(key)
                all_results.append(chunk)

    # ── Phase 1: search with source prefix (if requested) ──────────────
    if source_prefix:
        logger.info("Retriever: filtering by source prefix '%s'", source_prefix)
        for query in queries:
            _collect(_search(query, source_prefix, similarity_threshold))

        if all_results:
            logger.info(
                "Prefix search found %d chunks – skipping global search",
                len(all_results),
            )
        else:
            # Prefix returned nothing → fall back to searching all sources
            logger.info(
                "No results for prefix '%s' – falling back to all sources",
                source_prefix,
            )
            source_prefix = None   # clear prefix for phase 2

    # ── Phase 2: global search (no prefix, or prefix fallback) ─────────
    if not source_prefix:
        for query in queries:
            _collect(_search(query, None, similarity_threshold))

    # ── Phase 3: retry without threshold if still empty ────────────────
    if not all_results:
        logger.info(
            "No results with threshold %.2f – retrying without threshold",
            similarity_threshold,
        )
        for query in queries:
            _collect(_search(query, None, None))

    # ── Final: sort and trim ────────────────────────────────────────────
    all_results.sort(key=lambda c: c.get("score", 0), reverse=True)
    top_chunks = all_results[:final_top_k]

    logger.info(
        "Retriever: %d unique chunks from %d queries → returning top %d",
        len(all_results),
        len(queries),
        len(top_chunks),
    )
    return top_chunks
