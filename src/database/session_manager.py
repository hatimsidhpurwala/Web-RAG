import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

SESSION_FILE = Path("data/streamlit_sessions.json")
MAX_HISTORY = 10

def load_sessions() -> Dict[str, List[dict]]:
    """Load all sessions from the persistent JSON store."""
    if SESSION_FILE.exists():
        try:
            return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load sessions JSON: %s", e)
            return {}
    return {}

def save_sessions(sessions_dict: Dict[str, List[dict]]) -> None:
    """Save all sessions to the persistent JSON store."""
    try:
        SESSION_FILE.parent.mkdir(exist_ok=True, parents=True)
        SESSION_FILE.write_text(json.dumps(sessions_dict, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save sessions JSON: %s", e)

def get_session_history(session_id: str) -> List[dict]:
    """Get the conversation history for a specific session ID."""
    db = load_sessions()
    return db.get(session_id, [])

def update_session_history(session_id: str, history: List[dict]) -> None:
    """Update the conversation history for a specific session ID, keeping rolling window."""
    db = load_sessions()
    db[session_id] = history[-MAX_HISTORY:]
    save_sessions(db)

def clear_session(session_id: str) -> bool:
    """Delete a session's history. Returns True if deleted."""
    db = load_sessions()
    if session_id in db:
        del db[session_id]
        save_sessions(db)
        return True
    return False
