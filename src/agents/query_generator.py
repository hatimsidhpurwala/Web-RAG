"""
Query generator agent.

Converts a natural-language question into 2-4 optimised search queries
for the vector database. Uses with_structured_output for type safety.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import QueryGeneration

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a search-query optimisation engine for a RAG (Retrieval-Augmented
Generation) system. Your ONLY job is to convert a user question into 2–4
short, precise search queries that will find the right chunks in a vector DB.

QUERY CONSTRUCTION RULES:
1. Strip filler words: the, a, an, is, are, do, does, can, could, please, i.
2. Keep: nouns, proper nouns, technical terms, numbers, locations, verbs.
3. Each query MUST be 3–8 words. No full sentences.
4. No punctuation at end of queries.
5. Generate 2 queries minimum, 4 maximum.
6. Queries must be semantically diverse — do NOT generate near-duplicates.

QUERY TYPE — you MUST use exactly one of these values:
  "factual"     → general fact-based questions
  "comparative" → comparing two things
  "explanatory" → how/why questions
  "procedural"  → step-by-step / how-to questions
  "commercial"  → buy / price / where-to-buy / dealer / distributor / shop
  "contact"     → contact details / location / address / office / phone
  "document"    → questions about an uploaded PDF or file

QUERY GENERATION RULES BY TYPE:
  For "commercial" (buy, dealer, distributor, where to find, supplier, shop):
    - "[product model] buy online shop"
    - "[product] authorized distributor [location]"
    - "[product] reseller supplier dealer"
    - "[brand] official distributor [country]"

  For "contact" (contact info, address, office, phone, email):
    - "[company] [country/city] contact address"
    - "[company] distributor [location] email"
    - "[company] [country] office phone"
    - "[company] partner finder [region]"

  For "document" (what is the pdf, summarize, explain the file):
    - "product overview main features"
    - "document summary specifications"
    - "[product name] details"

  For all other types:
    - Generate the most specific, targeted queries possible.

DO NOT:
  - Generate questions (no "what is...?")
  - Generate full sentences
  - Repeat the same query twice
  - Use a query_type value not in the list above
  - Use "information" alone as a query

EXAMPLES:
  User: "what is the salto keycard chip?"
  query_type: "factual"
  Queries: ["SALTO keycard chip type", "MIFARE DESFire specifications", "SALTO CCVD chip security"]

  User: "contact details in india"
  query_type: "contact"
  Queries: ["SALTO India contact address", "SALTO distributor India", "SALTO Systems India office email"]

  User: "where can i buy SALTO keycards in UAE"
  query_type: "commercial"
  Queries: ["SALTO keycard buy UAE distributor", "SALTO CCVD20xx UAE reseller", "SALTO Systems UAE authorized dealer"]

  User: "what is the pdf about"
  query_type: "document"
  Queries: ["product overview main features", "document summary specifications", "company services solutions"]
"""

# LLM is now dynamically fetched inside the function to support UI multi-model selection.


# ── Public function ──────────────────────────────────────────────────────────

def generate_queries(
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> QueryGeneration:
    """Return optimised vector search queries for *question*.

    Returns a guaranteed valid QueryGeneration Pydantic object.
    """
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Include last 2 user turns for context (e.g. follow-up questions)
    if conversation_history:
        recent = [m for m in conversation_history[-6:] if m.get("role") == "user"]
        for msg in recent[-2:]:
            messages.append(HumanMessage(content=f"[context]: {msg.get('content', '')}"))

    messages.append(HumanMessage(content=question))

    try:
        from src.core.llm_factory import get_llm
        _llm = get_llm(temperature=0.1, max_tokens=300)
        _structured_llm = _llm.with_structured_output(QueryGeneration)
        result: QueryGeneration = _structured_llm.invoke(messages)
        # Safety: remove any empty or too-short queries
        result.queries = [q.strip() for q in result.queries if len(q.strip()) > 4]
        if not result.queries:
            result.queries = [question]
        logger.info("Generated %d queries: %s", len(result.queries), result.queries)
        return result
    except Exception as exc:
        logger.error("Query generation failed: %s – falling back to raw question", exc)
        return QueryGeneration(
            queries=[question],
            primary_entities=[],
            query_type="factual",
        )
