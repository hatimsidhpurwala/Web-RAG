"""
src/storage/session.py
Strict PostgreSQL storage for chat logs and document summaries.
"""

import os
import logging
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure config/.env is loaded before reading variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / "config" / ".env", override=True)

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    """Returns a direct PostgreSQL connection."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in config/.env")
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL at {DATABASE_URL}: {e}")
        raise ConnectionError(
            f"PostgreSQL connection failed. Ensure the server is running and "
            f"DATABASE_URL is correctly set. Error: {e}"
        )

def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Create chat history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create document summaries table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS doc_summaries (
                    session_id VARCHAR(50) NOT NULL,
                    doc_id VARCHAR(100) NOT NULL,
                    summary TEXT NOT NULL,
                    PRIMARY KEY (session_id, doc_id)
                );
            """)
            conn.commit()
        conn.close()
        logger.info("PostgreSQL database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error during PostgreSQL table initialization: {e}")

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Loads chat history for a session ID (phone number), keeping the last 20 messages."""
    history = []
    try:
        conn = get_connection()
        # Use RealDictCursor to return results as clean python dicts
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content FROM chat_history 
                WHERE session_id = %s 
                ORDER BY created_at ASC;
            """, (session_id,))
            rows = cur.fetchall()
            for r in rows:
                history.append({"role": r["role"], "content": r["content"]})
        conn.close()
    except Exception as e:
        logger.error(f"Error loading chat history from PostgreSQL: {e}")
    # Return last 20 messages to keep LLM context light
    return history[-20:]

def save_message(session_id: str, role: str, content: str) -> None:
    """Appends a new message to the chat log in PostgreSQL."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history (session_id, role, content) 
                VALUES (%s, %s, %s);
            """, (session_id, role, content))
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving chat message to PostgreSQL: {e}")

def clear_session(session_id: str) -> None:
    """Deletes all history and summaries for a specific session ID."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_history WHERE session_id = %s;", (session_id,))
            cur.execute("DELETE FROM doc_summaries WHERE session_id = %s;", (session_id,))
            conn.commit()
        conn.close()
        logger.info(f"Cleared session data for session: {session_id}")
    except Exception as e:
        logger.error(f"Error clearing session from PostgreSQL: {e}")

def save_doc_summary(session_id: str, doc_id: str, summary: str) -> None:
    """Saves or updates a document summary for a session ID."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO doc_summaries (session_id, doc_id, summary)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id, doc_id) 
                DO UPDATE SET summary = EXCLUDED.summary;
            """, (session_id, doc_id, summary))
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving document summary to PostgreSQL: {e}")

def get_doc_summary(session_id: str, doc_id: str) -> str:
    """Retrieves a document summary for a specific session and document ID."""
    summary = ""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT summary FROM doc_summaries 
                WHERE session_id = %s AND doc_id = %s;
            """, (session_id, doc_id))
            row = cur.fetchone()
            if row:
                summary = row[0]
        conn.close()
    except Exception as e:
        logger.error(f"Error loading document summary from PostgreSQL: {e}")
    return summary

# Auto-initialize tables when the module is imported
init_db()
