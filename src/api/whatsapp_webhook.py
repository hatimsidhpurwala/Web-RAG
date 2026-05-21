"""
WhatsApp webhook – FastAPI application that receives messages from
Twilio, routes them through the RAG agent, and returns TwiML responses.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Form, Response
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_graph import RAGAgent
from src.utils.assets import APP_DESCRIPTION, APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


app = FastAPI(title=APP_NAME, version=APP_VERSION, description=APP_DESCRIPTION)

# Lazy-initialised agent
_agent: RAGAgent | None = None


def _get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent


from src.database.session_manager import load_sessions as _load_sessions
from src.database.session_manager import save_sessions as _save_sessions
from src.database.session_manager import MAX_HISTORY as _MAX_HISTORY


@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "endpoints": {
            "/webhook": "POST – Twilio WhatsApp webhook",
            "/health": "GET – Health check",
        },
    }


@app.get("/health", tags=["info"])
async def health():
    """Simple health check."""
    return {"status": "ok"}


@app.post("/webhook", tags=["whatsapp"])
async def whatsapp_webhook(
    From: str = Form(""),
    Body: str = Form(""),
):
    """Receive a WhatsApp message from Twilio, process it, and return a
    TwiML response.
    """
    user_number = From.replace("whatsapp:", "").strip()
    message = Body.strip()

    logger.info("WhatsApp message from %s: %s", user_number, message[:80])

    if not message:
        return _twiml_response("I didn't receive a message. Please try again.")

    # ── Handle Reset Command ──
    if message.lower() in ["reset", "clear", "restart"]:
        try:
            sessions_db = _load_sessions()
            if user_number in sessions_db:
                del sessions_db[user_number]
                _save_sessions(sessions_db)
            return _twiml_response("🧹 Your conversation history has been cleared! How can I help you today?")
        except Exception as exc:
            logger.error("Failed to clear session: %s", exc)
            return _twiml_response("Failed to clear history. Please try again.")

    # ── Load Persistent Conversation History ──
    sessions_db = _load_sessions()
    history = sessions_db.get(user_number, [])

    try:
        agent = _get_agent()
        # Pass the phone number as the session_id so LangGraph isolates the thread
        result = agent.ask(message, conversation_history=history, session_id=user_number)
        answer = result.get("final_answer", "Sorry, I couldn't process that.")

        # Append sources if available
        sources = result.get("sources", [])
        if sources:
            answer += "\n\n📚 Sources:\n" + "\n".join(f"• {s}" for s in sources[:3])

        if result.get("web_search_performed"):
            answer = "🌐 (web research performed)\n\n" + answer

    except Exception as exc:
        logger.error("Error processing WhatsApp message: %s", exc, exc_info=True)
        answer = "I ran into an error processing your message. Please try again."

    # ── Update and Save Persistent History ──
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    # Enforce rolling window memory
    sessions_db[user_number] = history[-_MAX_HISTORY:]
    _save_sessions(sessions_db)

    return _twiml_response(answer)


# ======================================================================
# Helpers
# ======================================================================

def _twiml_response(body: str) -> Response:
    """Return a Twilio-compatible TwiML XML response."""
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{_escape_xml(body)}</Message>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
