"""
src/storage/session.py
Chat history and doc summaries per user, stored in a single JSON file.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

SESSION_FILE = Path("data/sessions.json")
MAX_HISTORY = 20

def _load() -> Dict[str, Any]:
    if SESSION_FILE.exists():
        try: return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def _save(db: Dict[str, Any]) -> None:
    SESSION_FILE.parent.mkdir(exist_ok=True, parents=True)
    SESSION_FILE.write_text(json.dumps(db, indent=2), encoding="utf-8")

def _get_session(session_id: str, db: Dict[str, Any]) -> Dict[str, Any]:
    if session_id not in db:
        db[session_id] = {"history": [], "doc_summaries": {}}
    return db[session_id]

def get_history(session_id: str) -> List[dict]:
    db = _load()
    return _get_session(session_id, db).get("history", [])

def save_message(session_id: str, role: str, content: str) -> None:
    db = _load()
    sess = _get_session(session_id, db)
    sess["history"].append({"role": role, "content": content})
    sess["history"] = sess["history"][-MAX_HISTORY:]
    _save(db)

def clear_session(session_id: str) -> None:
    db = _load()
    if session_id in db:
        del db[session_id]
        _save(db)

def save_doc_summary(session_id: str, doc_id: str, summary: str) -> None:
    db = _load()
    sess = _get_session(session_id, db)
    if "doc_summaries" not in sess: sess["doc_summaries"] = {}
    sess["doc_summaries"][doc_id] = summary
    _save(db)

def get_doc_summary(session_id: str, doc_id: str) -> str:
    db = _load()
    sess = _get_session(session_id, db)
    return sess.get("doc_summaries", {}).get(doc_id, "")
