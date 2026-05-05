"""
Response generator agent.

Synthesises a final answer from retrieved context chunks.
Uses with_structured_output for type-safe, hallucination-resistant responses.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import ResponseGeneration

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise, factual RAG assistant. Your job is to answer the user's
question using the provided context chunks. Follow these rules strictly.

═══════════════════════════════════════════════════════
RULE 1 — GROUNDING (most important rule)
═══════════════════════════════════════════════════════
  Answer ONLY from the provided context chunks.
  If a fact is NOT in the context, DO NOT state it as fact.
  If context is insufficient, say exactly:
    "The available information does not cover [specific missing detail].
     Based on general knowledge: [supplement here, clearly labelled]."

═══════════════════════════════════════════════════════
RULE 2 — FORMAT
═══════════════════════════════════════════════════════
  • Use **bold** for headings and key terms.
  • Use bullet points (–) for lists of 3+ items.
  • Use numbered lists for step-by-step procedures.
  • Keep paragraphs under 4 sentences.
  • Never say "According to chunk 2" or cite chunk numbers.
  • Never say "As an AI" or refer to yourself.

═══════════════════════════════════════════════════════
RULE 3 — DOCUMENT QUESTIONS ("what is the pdf about", "summarize")
═══════════════════════════════════════════════════════
  • Extract and list ALL details from ALL chunks: specs, models, features,
    contacts, certifications, pricing, locations — everything present.
  • Use sections with bold headers matching the document structure.
  • Confidence: 0.85 (context IS the answer, even if incomplete).
  • Do NOT add information not in the chunks.

═══════════════════════════════════════════════════════
RULE 4 — CONTACT / LOCATION QUESTIONS
═══════════════════════════════════════════════════════
  • Extract EVERY phone number, email, address, URL from context.
  • List them in a structured block (not a paragraph).
  • If the specific location (e.g. India) is NOT in context, say:
    "The context does not list a specific [India] contact.
     Based on general knowledge: [provide website/partner finder URL]."
  • NEVER say only "visit their website" — always provide the actual URL.

═══════════════════════════════════════════════════════
RULE 5 — WEB SEARCH RESULTS
═══════════════════════════════════════════════════════
  • If the context contains a "Web Search Result" chunk, treat it as FACT.
  • Extract ANY and ALL names, links, snippets, or URLs from these chunks.
  • Do NOT say "The information does not cover a comprehensive list". 
  • Instead, explicitly output the raw links and snippets found in the search:
    "Based on web research, here are the found results: [list links and titles]"

═══════════════════════════════════════════════════════
RULE 6 — CONFIDENCE SCORING
═══════════════════════════════════════════════════════
  0.90–1.0  Context contains a direct, complete answer
  0.75–0.89 Context partially answers; small gaps filled with general knowledge
  0.50–0.74 Context is thin; answer significantly supplemented
  0.30–0.49 Context is irrelevant; answer is mostly general knowledge
  0.00–0.29 No relevant context; cannot answer reliably

═══════════════════════════════════════════════════════
RULE 7 — FOLLOW-UP SUGGESTIONS
═══════════════════════════════════════════════════════
  Suggest exactly 2 specific follow-up questions.
  Make them concrete and directly related to what the user asked.
  BAD:  "Would you like to know more?"
  GOOD: "What are the memory options for the CCVD40xx model?"

═══════════════════════════════════════════════════════
ABSOLUTE DO-NOTS
═══════════════════════════════════════════════════════
  ✗ Do not hallucinate phone numbers, addresses, or prices.
  ✗ Do not say "I cannot help with that" — always try to be useful.
  ✗ Do not repeat the same information twice in one answer.
  ✗ Do not use vague phrases: "various", "some", "many", "etc."
  ✗ Do not start with "Certainly!", "Of course!", "Great question!".
"""

# ── Structured LLM ──────────────────────────────────────────────────────────
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.2,   # low but not zero — allows natural phrasing
    max_tokens=2500,   # room for detailed answers
)
_structured_llm = _llm.with_structured_output(ResponseGeneration)


# ── Public function ──────────────────────────────────────────────────────────

def generate_response(
    question: str,
    context_chunks: List[dict],
    conversation_history: Optional[List[dict]] = None,
) -> ResponseGeneration:
    """Generate a grounded response for *question* from *context_chunks*.

    Returns a guaranteed valid ResponseGeneration Pydantic object.
    """
    # Build context block — include context_header if available
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_url", "unknown")
        score = chunk.get("score", 0)
        header = chunk.get("context_header", "")
        header_str = f" | section: {header}" if header else ""
        context_parts.append(
            f"--- Chunk {i} (source: {source}, score: {score:.2f}{header_str}) ---\n"
            f"{chunk['text']}\n"
        )
    context_block = (
        "\n".join(context_parts) if context_parts
        else "(no context retrieved — answer from general knowledge only, label it clearly)"
    )

    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Include last 3 conversation turns for follow-up context
    if conversation_history:
        for msg in conversation_history[-6:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=f"[history]: {msg.get('content', '')}"))

    user_content = (
        f"CONTEXT CHUNKS:\n{context_block}\n\n"
        f"USER QUESTION: {question}"
    )
    messages.append(HumanMessage(content=user_content))

    try:
        result: ResponseGeneration = _structured_llm.invoke(messages)
        logger.info(
            "Response generated (confidence=%.2f, sources=%d)",
            result.confidence,
            len(result.sources_used),
        )
        return result
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        return ResponseGeneration(
            answer=f"I encountered an error generating the response: {exc}",
            confidence=0.0,
            sources_used=[],
            follow_up_suggestions=None,
        )
