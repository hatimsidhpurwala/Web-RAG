"""
Direct responder agent – handles non-retrieval intents such as
greetings, farewells, gratitude, and general-knowledge questions.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a friendly, professional AI assistant. The user's message does
NOT require a knowledge-base lookup. Respond naturally and helpfully.

For greetings: respond warmly and ask how you can help.
For gratitude: acknowledge politely.
For farewells: say goodbye warmly.
For general knowledge: provide a concise, accurate answer.
For clarification: ask the user to rephrase their question clearly.

ABSOLUTE RULES:
- NEVER say "I didn't receive any PDF" or claim a file was not shared.
- NEVER say "I made a mistake earlier" or contradict the conversation history.
- If the user seems to be following up on a document question, simply ask:
  "Could you clarify what you'd like to know about the document?"
- Keep your response concise (1-3 sentences). Be professional but warm.

Respond with valid JSON:
{
  "answer": "<your response>"
}
"""


def direct_response(
    question: str,
    intent: str,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """Generate a direct (non-retrieval) response for *question*."""
    client = Groq(api_key=GROQ_API_KEY)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": f"[Intent: {intent}]\n{question}",
        }
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        answer = data.get("answer", raw)
        logger.info("Direct response for intent '%s'", intent)
        return answer
    except Exception as exc:
        logger.error("Direct response failed: %s", exc)
        return "I'm here to help! Could you rephrase your question?"
