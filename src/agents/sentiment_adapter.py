"""
Sentiment-Aware Response Adapter – detects user emotion and urgency,
then adapts the response tone accordingly.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_DETECT_PROMPT = """\
Analyze the user's message for emotional state and urgency.

Return JSON:
{
  "emotion": "neutral|frustrated|confused|satisfied|excited|urgent|curious",
  "urgency": "low|medium|high",
  "politeness": "casual|polite|formal|curt",
  "tone_hint": "Brief instruction on how to adapt the response tone"
}

Signals:
- frustrated: "I already asked", "this doesn't work", caps, "!!!"
- confused: "I don't understand", "what do you mean", "?"
- urgent: "ASAP", "urgent", "quickly", "immediately", "need now"
- excited: "amazing!", "great!", "love"
- curious: questions, "tell me more", "how does"
"""


class SentimentAdapter:
    """Detect user sentiment and adapt response tone."""

    def analyze_sentiment(
        self,
        message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> Dict[str, str]:
        """Detect emotion, urgency, and politeness level."""
        # Fast-path heuristics
        lower = message.lower()
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)

        if caps_ratio > 0.5 and len(message) > 10:
            return {
                "emotion": "frustrated",
                "urgency": "high",
                "politeness": "curt",
                "tone_hint": "Be concise and solution-focused. Acknowledge urgency.",
            }

        if any(w in lower for w in ["asap", "urgent", "immediately", "urgent response"]):
            return {
                "emotion": "urgent",
                "urgency": "high",
                "politeness": "casual",
                "tone_hint": "Lead with the key answer immediately. Details after.",
            }

        if any(w in lower for w in ["don't understand", "confused", "what do you mean"]):
            return {
                "emotion": "confused",
                "urgency": "medium",
                "politeness": "polite",
                "tone_hint": "Use simpler language. Add examples. Break down concepts.",
            }

        if any(w in lower for w in ["thanks", "thank you", "great", "perfect", "amazing"]):
            return {
                "emotion": "satisfied",
                "urgency": "low",
                "politeness": "polite",
                "tone_hint": "Warm and helpful. Offer further assistance.",
            }

        # Default
        return {
            "emotion": "neutral",
            "urgency": "low",
            "politeness": "polite",
            "tone_hint": "Standard professional and helpful tone.",
        }

    def adapt_response(
        self,
        answer: str,
        sentiment: Dict[str, str],
    ) -> str:
        """Adapt *answer* based on detected *sentiment*."""
        emotion = sentiment.get("emotion", "neutral")
        urgency = sentiment.get("urgency", "low")

        if emotion == "frustrated":
            # Concise, solution-focused, empathetic opening
            prefix = "I understand. Here's what you need:\n\n"
            # Trim to essentials
            lines = answer.split("\n")
            key_lines = [l for l in lines if l.strip()][:8]
            return prefix + "\n".join(key_lines)

        elif emotion == "confused":
            prefix = "Let me break this down simply:\n\n"
            return prefix + answer

        elif urgency == "high":
            # Instead of destroying Markdown by removing newlines, 
            # just add a priority header.
            prefix = "⏱️ **Priority Response:**\n\n"
            return prefix + answer

        elif emotion == "satisfied":
            suffix = "\n\nLet me know if you need anything else! 😊"
            return answer + suffix

        return answer
