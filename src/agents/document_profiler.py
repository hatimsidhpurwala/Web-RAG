import json
import logging
from typing import Dict
from groq import Groq
from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_PROFILER_PROMPT = """\
You are an expert Document Profiler. Analyze the provided text from a newly uploaded document.
Provide a concise summary of what this document is about, and list its core topics.
Keep the summary under 3 sentences.

Return EXACTLY this JSON format:
{
  "summary": "Brief summary...",
  "topics": ["topic1", "topic2", "topic3"]
}
"""

def profile_document(text: str) -> Dict:
    """Analyze text and return a profile."""
    client = Groq(api_key=GROQ_API_KEY)
    
    # We only need the first 4000 characters to get a good sense of the document
    sample_text = text[:4000]
    
    messages = [
        {"role": "system", "content": _PROFILER_PROMPT},
        {"role": "user", "content": f"Document Text:\n{sample_text}"}
    ]
    
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data
    except Exception as exc:
        logger.error("Failed to profile document: %s", exc)
        return {"summary": "Unknown document.", "topics": []}
