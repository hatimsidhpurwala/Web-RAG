"""
Intent classification agent.

Uses LangChain's with_structured_output() — the Pydantic model IS the schema.
Groq is forced via tool-calling to always return the exact structure.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import IntentClassification

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an intent-classification engine. Classify the user message into
EXACTLY ONE of these categories. Never invent new categories.

CATEGORIES (read carefully before choosing):
  greeting        – Pure social openers: "hi", "hello", "hey", "good morning"
  gratitude       – Thank-you expressions: "thanks", "thank you", "appreciate it"
  farewell        – Closing messages: "bye", "goodbye", "see you", "take care"
  clarification   – Asking for MORE DETAIL on the PREVIOUS assistant reply,
                    e.g. "explain more", "what do you mean", "can you elaborate"
  general_knowledge – Factual questions whose answer does NOT require searching
                    any uploaded file or scraped website, e.g. "what is Python",
                    "who is Einstein", "how does GPS work"
  retrieval_needed  – EVERYTHING ELSE that requires searching the knowledge base:
                    product specs, prices, company info, uploaded PDF content,
                    contact details, "where can I buy", "what does the doc say"

STRICT RULES — follow in this exact order:
1. If the message mentions "pdf", "document", "file", "uploaded", "datasheet",
   OR references a named company/product/service → ALWAYS "retrieval_needed".
2. If the message is a FOLLOW-UP question about a topic from history that
   needed retrieval → classify as "retrieval_needed".
3. If the message is purely social (no information request) → greeting/gratitude/farewell.
4. If the message asks "explain more" or "what did you mean" → "clarification".
5. Only use "general_knowledge" for standalone factual questions with NO company
   or product context.

CONFIDENCE RULES:
  • 0.95–1.0 = Crystal clear (pure greeting, direct product question)
  • 0.80–0.94 = Clear but slightly ambiguous
  • 0.60–0.79 = Ambiguous — could fit multiple categories
  • Below 0.60 = Very uncertain — default to "retrieval_needed"

DO NOT:
  - Mix categories (pick exactly one)
  - Set confidence above 0.95 unless you are 100% certain
  - Classify product/company questions as "general_knowledge"
"""

# LLM is now dynamically fetched inside the function to support UI multi-model selection.


# ── Public function ──────────────────────────────────────────────────────────

def classify_intent(
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> IntentClassification:
    """Classify *question* and return a structured IntentClassification.

    Returns a guaranteed valid Pydantic object — no json.loads risk.
    """
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Inject recent history so clarification detection works
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=f"[prev user]: {content}"))
            elif role == "assistant":
                messages.append(HumanMessage(content=f"[prev assistant]: {content[:200]}"))

    messages.append(HumanMessage(content=f"[current]: {question}"))

    try:
        from src.core.llm_factory import get_llm
        _llm = get_llm(temperature=0.0, max_tokens=200)
        _structured_llm = _llm.with_structured_output(IntentClassification)
        result: IntentClassification = _structured_llm.invoke(messages)
        logger.info(
            "Intent: %s (confidence=%.2f, retrieval=%s)",
            result.intent,
            result.confidence,
            result.needs_retrieval,
        )
        return result
    except Exception as exc:
        logger.error("Intent classification failed: %s – defaulting to retrieval", exc)
        return IntentClassification(
            intent="retrieval_needed",
            confidence=0.5,
            reasoning=f"Classification error: {exc}",
            needs_retrieval=True,
        )
