"""
State schema for the LangGraph agent pipeline.

Defines AgentState – the single shared dictionary that flows
between every node in the graph. Every field is optional (total=False)
so nodes only need to return the keys they actually update.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    # ── Session isolation ───────────────────────────────────────────────
    session_id: str             # unique UUID per browser tab / user session

    # ── Core pipeline fields ────────────────────────────────────────────
    question: str               # the (possibly enhanced) working question
    original_question: str      # raw user input before translation / rewrite
    intent: str                 # classified intent label
    intent_confidence: float    # how confident the classifier is
    needs_retrieval: bool       # True  → go to retrieval pipeline
    generated_queries: List[str]  # optimised search queries
    retrieved_chunks: List[dict]  # chunks returned from vector store
    final_answer: str           # assembled answer text
    confidence: float           # answer confidence 0–1
    sources: List[str]          # source URLs used
    follow_up_suggestions: List[str]
    conversation_history: List[dict]   # previous turns
    web_search_performed: bool
    research_info: Dict[str, Any]      # metadata from web research

    # ── Enhanced / advanced fields ──────────────────────────────────────
    query_strategy: Dict[str, Any]     # output of query_analyzer
    reasoning_chain: Dict[str, Any]    # chain-of-thought steps
    fact_check_report: Dict[str, Any]  # per-claim verification results
    sentiment: Dict[str, str]          # detected emotion / urgency
    conversation_state: Dict[str, Any] # topic, state, enhanced_query
    clarification_needed: bool
    clarification_data: Dict[str, Any]
    language_info: Dict[str, Any]      # source lang, needs_translation
    quality_score: float               # critic agent score
    response_time_ms: int
    enhanced_features_used: List[str]  # audit trail of features activated
    source_prefix: str                 # e.g. "pdf_" – restricts retrieval to that source type
    active_doc_sites: List[str]        # List of site_names for currently active documents
    raw_search_results: List[dict]     # DuckDuckGo result cards {title, url, snippet}
