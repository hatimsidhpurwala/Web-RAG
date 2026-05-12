import json
import logging
from typing import Dict
from config.settings import LLM_MODEL

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
    from src.core.llm_factory import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # We only need the first 4000 characters to get a good sense of the document
    sample_text = text[:4000]
    
    messages = [
        SystemMessage(content=_PROFILER_PROMPT),
        HumanMessage(content=f"Document Text:\n{sample_text}")
    ]
    
    try:
        llm = get_llm(temperature=0.1, max_tokens=300)
        resp = llm.invoke(messages)
        
        # Parse JSON, handling potential markdown code blocks
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        data = json.loads(content)
        return data
    except Exception as exc:
        logger.error("Failed to profile document: %s", exc)
        return {"summary": "Unknown document.", "topics": []}
