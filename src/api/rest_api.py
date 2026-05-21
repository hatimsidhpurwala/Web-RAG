"""
src/api/rest_api.py

Universal REST API — FastAPI server that exposes the full RAG pipeline.

Endpoints
---------
POST /api/upload          Upload a file (PDF/image/audio/doc) → index + optional question
POST /api/ask             Ask a text question tied to a session
POST /api/voice           Upload audio → transcribe → ask
POST /api/ocr             Upload image → OCR → ask
GET  /api/session/{id}    Session status & metadata
DELETE /api/session/{id}  Wipe all session data (Qdrant + disk)
GET  /api/health          Health check for all connected services

Run alongside Streamlit:
    uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import (
    COLLECTION_NAME,
    GROQ_API_KEY,
    TESSERACT_CMD,
    WHISPER_MODEL,
)
from src.agents.graph.agent import RAGAgent
from src.core.chunker import chunk_markdown
from src.core.cleaner import deduplicate_chunks, normalize_text
from src.core.embedder import embed_chunks
from src.database.vector_store import VectorStore

import json

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# Persistent Sessions
# ══════════════════════════════════════════════════════════════════════
from src.database.session_manager import load_sessions as _load_sessions
from src.database.session_manager import save_sessions as _save_sessions
from src.database.session_manager import MAX_HISTORY as _MAX_HISTORY


# ══════════════════════════════════════════════════════════════════════
# App bootstrap
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Smart RAG Universal API",
    description=(
        "Hybrid RAG pipeline exposed as a REST API. "
        "Accepts files, text, audio, and images. "
        "Returns structured JSON answers with sources, confidence, and follow-ups."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins so React / Vue / mobile apps / bots can call freely.
# Tighten `allow_origins` in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════
# Shared singletons (lazy-initialised)
# ══════════════════════════════════════════════════════════════════════

_vector_store: Optional[VectorStore] = None
_agents: Dict[str, RAGAgent] = {}          # session_id → RAGAgent
_session_meta: Dict[str, dict] = {}        # session_id → metadata dict


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_agent(session_id: str) -> RAGAgent:
    """Return (or create) a RAGAgent for this session."""
    if session_id not in _agents:
        _agents[session_id] = RAGAgent(get_vector_store())
        logger.info("Created new RAGAgent for session %s", session_id)
    return _agents[session_id]


def _touch_session(session_id: str) -> None:
    """Create session metadata record if it doesn't exist."""
    if session_id not in _session_meta:
        _session_meta[session_id] = {
            "session_id":    session_id,
            "created_at":    datetime.utcnow().isoformat() + "Z",
            "active_doc_sites": [],
            "questions_asked":  0,
            "chunks_indexed":   0,
        }


def _ensure_session(session_id: Optional[str]) -> str:
    """Return existing session or create a new UUID."""
    sid = session_id or str(uuid.uuid4())
    _touch_session(sid)
    return sid


# ══════════════════════════════════════════════════════════════════════
# Pydantic response schemas
# ══════════════════════════════════════════════════════════════════════

class AnswerResponse(BaseModel):
    session_id:   str
    question:     str
    answer:       str
    confidence:   float
    sources:      List[str]           = Field(default_factory=list)
    follow_ups:   List[str]           = Field(default_factory=list)
    web_searched: bool                = False
    features:     List[str]           = Field(default_factory=list)
    response_ms:  int                 = 0
    fact_check:   Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    session_id:    str
    filename:      str
    file_type:     str
    chunks_indexed: int
    from_cache:    bool               = False
    immediate_answer: Optional[AnswerResponse] = None


class SessionStatusResponse(BaseModel):
    session_id:        str
    created_at:        str
    active_doc_sites:  List[str]
    questions_asked:   int
    chunks_indexed:    int


class HealthResponse(BaseModel):
    status:   str
    qdrant:   str
    groq:     str
    gemini:   str
    openai:   str
    anthropic: str
    timestamp: str


# ══════════════════════════════════════════════════════════════════════
# Extraction helpers
# ══════════════════════════════════════════════════════════════════════
from src.core.document_parser import (
    extract_text_from_file as _extract_text_from_file,
    extract_image_ocr as _extract_image_ocr,
    transcribe_audio as _transcribe_audio
)

def _safe_extract_text_from_file(raw: bytes, filename: str) -> tuple[str, str]:
    try:
        return _extract_text_from_file(raw, filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e),
        )


def _index_text(text: str, site_name: str, vs: VectorStore) -> int:
    """Chunk → deduplicate → embed → store. Returns chunk count."""
    text = normalize_text(text)
    chunks = chunk_markdown(text, source_url=site_name)
    chunks = deduplicate_chunks(chunks)
    chunks = embed_chunks(chunks, show_progress=False)
    return vs.store_chunks_for_site(chunks, site_name)


def _run_agent(
    question: str,
    session_id: str,
    active_doc_sites: List[str],
    conversation_history: Optional[List[dict]] = None,
) -> AnswerResponse:
    """Run the full RAG agent and return a structured AnswerResponse."""
    agent = get_agent(session_id)
    
    # ── Load Persistent Conversation History ──
    sessions_db = _load_sessions()
    history = sessions_db.get(session_id, [])
    
    # Fallback to client-provided history if the server has no record yet
    if not history and conversation_history:
        history = conversation_history
        
    result = agent.ask(
        question,
        conversation_history=history,
        session_id=session_id,
        active_doc_sites=active_doc_sites,
    )
    _session_meta[session_id]["questions_asked"] += 1
    
    answer = result.get("final_answer", "")
    
    # ── Update and Save Persistent History ──
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    sessions_db[session_id] = history[-_MAX_HISTORY:]
    _save_sessions(sessions_db)
    return AnswerResponse(
        session_id=session_id,
        question=question,
        answer=result.get("final_answer", ""),
        confidence=result.get("confidence", 0.0),
        sources=result.get("sources", []),
        follow_ups=result.get("follow_up_suggestions", []),
        web_searched=result.get("web_search_performed", False),
        features=result.get("enhanced_features_used", []),
        response_ms=result.get("response_time_ms", 0),
        fact_check=result.get("fact_check_report"),
    )


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — File Upload + Index (+ optional immediate question)
# ══════════════════════════════════════════════════════════════════════

@app.post(
    "/api/upload",
    response_model=UploadResponse,
    summary="Upload a file and index its content",
    tags=["Core"],
)
async def upload_file(
    file: UploadFile = File(..., description="PDF, Word, Excel, PPT, image, CSV, or text file"),
    session_id: Optional[str] = Form(None, description="Existing session ID. Auto-generated if omitted."),
    question: Optional[str]   = Form(None, description="Optional question to answer immediately after indexing"),
):
    """
    Upload any supported file. The API extracts its text, chunks it,
    embeds it, and stores it in Qdrant under this session's namespace.

    If **question** is also provided, the pipeline answers it immediately
    and includes the answer in the response.

    Supported formats: PDF, DOCX, XLSX/CSV, PPTX, PNG/JPG/BMP/TIFF, TXT, MD.
    """
    sid = _ensure_session(session_id)
    vs  = get_vector_store()

    raw      = await file.read()
    filename = file.filename or "upload"
    file_hash = hashlib.sha256(raw).hexdigest()[:16]

    try:
        text, category = _safe_extract_text_from_file(raw, filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="File produced no extractable text.")

    # Namespace site_name under this user's session
    site_name   = f"pdf_{sid}_{file_hash}"
    from_cache  = vs.has_site(site_name)
    chunks_stored = 0

    if not from_cache:
        try:
            vs.clear_site(site_name)
            chunks_stored = _index_text(text, site_name, vs)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")
    else:
        # Count existing chunks for reporting
        try:
            info = vs.get_collection_info()
            chunks_stored = info.get("points_count", 0)
        except Exception:
            chunks_stored = 0

    # Update session metadata
    meta = _session_meta[sid]
    if site_name not in meta["active_doc_sites"]:
        meta["active_doc_sites"].append(site_name)
    meta["chunks_indexed"] += chunks_stored

    # Document profiling (non-blocking — best effort)
    try:
        from src.agents.document_profiler import profile_document
        from src.database.metadata_registry import MetadataRegistry
        profile = profile_document(text)
        profile["original_filename"] = filename
        MetadataRegistry().save_profile(site_name, profile)
    except Exception:
        pass

    # Optional immediate answer
    immediate: Optional[AnswerResponse] = None
    if question and question.strip():
        try:
            immediate = _run_agent(
                question.strip(),
                sid,
                meta["active_doc_sites"],
            )
        except Exception as exc:
            logger.error("Immediate answer failed: %s", exc)

    return UploadResponse(
        session_id=sid,
        filename=filename,
        file_type=category,
        chunks_indexed=chunks_stored,
        from_cache=from_cache,
        immediate_answer=immediate,
    )


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Ask a Text Question
# ══════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question:            str            = Field(..., min_length=1)
    session_id:          Optional[str]  = Field(None, description="Session to query. Auto-created if omitted.")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description='Previous turns as [{"role": "user"|"assistant", "content": "..."}]'
    )


@app.post(
    "/api/ask",
    response_model=AnswerResponse,
    summary="Ask a text question",
    tags=["Core"],
)
async def ask_question(body: AskRequest):
    """
    Send a natural-language question. The agent runs the full pipeline:
    intent classification → query optimisation → vector retrieval →
    optional web research → response generation → fact verification.

    Returns a structured answer with confidence score, sources, and follow-ups.
    """
    sid = _ensure_session(body.session_id)
    meta = _session_meta[sid]
    return _run_agent(
        body.question.strip(),
        sid,
        meta["active_doc_sites"],
        body.conversation_history,
    )


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Voice (Audio → Transcribe → Ask)
# ══════════════════════════════════════════════════════════════════════

@app.post(
    "/api/voice",
    response_model=AnswerResponse,
    summary="Upload audio → transcribe via Whisper → answer",
    tags=["Multimodal"],
)
async def voice_question(
    audio: UploadFile = File(..., description="WAV, MP3, M4A, OGG, or WEBM audio file"),
    session_id: Optional[str] = Form(None),
):
    """
    Upload an audio file. The API transcribes it with Groq Whisper,
    then runs the transcribed text through the full RAG pipeline.

    The `question` field in the response will contain the transcribed text.
    """
    sid = _ensure_session(session_id)
    raw    = await audio.read()
    suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"

    try:
        transcription = _transcribe_audio(raw, suffix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    if not transcription.strip():
        raise HTTPException(status_code=422, detail="Audio produced no transcription.")

    meta = _session_meta[sid]
    return _run_agent(transcription, sid, meta["active_doc_sites"])


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — Image OCR → Ask
# ══════════════════════════════════════════════════════════════════════

class OcrAnswerResponse(AnswerResponse):
    extracted_text: str = ""


@app.post(
    "/api/ocr",
    response_model=OcrAnswerResponse,
    summary="Upload image → OCR → answer",
    tags=["Multimodal"],
)
async def ocr_question(
    image: UploadFile = File(..., description="PNG, JPG, BMP, TIFF, or WEBP image"),
    session_id: Optional[str] = Form(None),
):
    """
    Upload an image. The API runs Tesseract OCR to extract any visible text,
    then runs that text through the full RAG pipeline as a question.

    The `extracted_text` field in the response contains the raw OCR output.
    """
    sid = _ensure_session(session_id)
    raw = await image.read()

    try:
        ocr_text = _extract_image_ocr(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    if not ocr_text.strip():
        raise HTTPException(status_code=422, detail="Image contained no readable text.")

    meta   = _session_meta[sid]
    result = _run_agent(ocr_text, sid, meta["active_doc_sites"])
    return OcrAnswerResponse(extracted_text=ocr_text, **result.model_dump())


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Session Status
# ══════════════════════════════════════════════════════════════════════

@app.get(
    "/api/session/{session_id}",
    response_model=SessionStatusResponse,
    summary="Get session metadata",
    tags=["Session"],
)
async def get_session(session_id: str):
    """
    Returns metadata for the given session:
    - When it was created
    - Which document sources have been indexed
    - Total chunks stored
    - How many questions have been asked
    """
    if session_id not in _session_meta:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    meta = _session_meta[session_id]
    return SessionStatusResponse(**meta)


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — Delete Session
# ══════════════════════════════════════════════════════════════════════

@app.delete(
    "/api/session/{session_id}",
    summary="Delete all data for a session",
    tags=["Session"],
)
async def delete_session(session_id: str):
    """
    Wipes everything for this session:
    - Qdrant vectors (all namespaced site_names)
    - In-memory agent and metadata
    - LangGraph MemorySaver thread checkpoint

    After this call the session_id is invalid.
    """
    if session_id not in _session_meta:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    vs   = get_vector_store()
    meta = _session_meta[session_id]
    deleted_sites = []

    # Remove all Qdrant namespaced data for this session
    for site_name in meta.get("active_doc_sites", []):
        try:
            vs.clear_site(site_name)
            deleted_sites.append(site_name)
        except Exception as exc:
            logger.warning("Could not clear site '%s': %s", site_name, exc)

    # Also remove any web-search namespaced chunks (prefix = "{session_id}:")
    # by scrolling and collecting matching site names
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchText
        scroll_results, _ = vs.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=["site_name"],
            with_vectors=False,
        )
        web_sites = set(
            p.payload["site_name"]
            for p in scroll_results
            if p.payload.get("site_name", "").startswith(f"{session_id}:")
        )
        for site in web_sites:
            vs.clear_site(site)
            deleted_sites.append(site)
    except Exception as exc:
        logger.warning("Web-chunk cleanup error for session %s: %s", session_id, exc)

    # Drop agent instance
    _agents.pop(session_id, None)
    # Drop metadata
    del _session_meta[session_id]
    
    # Drop from JSON persistent memory
    try:
        from src.database.session_manager import clear_session
        clear_session(session_id)
    except Exception as exc:
        logger.warning("JSON memory cleanup error for session %s: %s", session_id, exc)

    return {
        "deleted":       True,
        "session_id":    session_id,
        "sites_removed": deleted_sites,
    }

from fastapi import Request

@app.post(
    "/api/webhook/universal",
    summary="Universal Webhook for Telegram, Instagram, Twilio, or generic JSON",
    tags=["Core"],
)
async def universal_webhook(request: Request):
    """
    A single universal endpoint you can paste into ANY platform (Telegram, Instagram, Twilio, Make.com).
    It auto-detects the payload format, extracts the user ID and message, and responds accordingly.
    """
    content_type = request.headers.get("content-type", "")
    session_id = "default_user"
    message = ""

    if "application/x-www-form-urlencoded" in content_type:
        # Likely Twilio WhatsApp
        form_data = await request.form()
        session_id = form_data.get("From", "").replace("whatsapp:", "").strip()
        message = form_data.get("Body", "").strip()
    elif "application/json" in content_type:
        json_data = await request.json()
        
        # Telegram detection
        if "message" in json_data and isinstance(json_data["message"], dict) and "chat" in json_data["message"]:
            session_id = str(json_data["message"]["chat"].get("id", ""))
            message = json_data["message"].get("text", "").strip()
        # Instagram/Messenger detection (Facebook Graph API)
        elif "entry" in json_data and isinstance(json_data["entry"], list):
            entry = json_data["entry"][0]
            if "messaging" in entry:
                msg_event = entry["messaging"][0]
                session_id = msg_event.get("sender", {}).get("id", "")
                message = msg_event.get("message", {}).get("text", "").strip()
        # Generic JSON detection
        else:
            session_id = json_data.get("session_id", json_data.get("user_id", "default_user"))
            message = json_data.get("message", json_data.get("text", json_data.get("question", "")))

    if not message:
        return JSONResponse(content={"error": "No message found in payload"}, status_code=400)

    # Use the existing Ask pipeline
    sid = _ensure_session(session_id)
    meta = _session_meta[sid]
    
    try:
        from src.database.session_manager import load_sessions, save_sessions, MAX_HISTORY
        sessions_db = load_sessions()
        history = sessions_db.get(sid, [])
        
        result = _run_agent(message, sid, meta["active_doc_sites"], conversation_history=history)
        answer = result.answer
        
        # Update persistent history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        sessions_db[sid] = history[-MAX_HISTORY:]
        save_sessions(sessions_db)
        
        return JSONResponse(content={"session_id": sid, "reply": answer})
    except Exception as exc:
        logger.error("Universal webhook failed: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════
# ENDPOINT 7 — Health Check
# ══════════════════════════════════════════════════════════════════════

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check for all connected services",
    tags=["Monitoring"],
)
async def health_check():
    """
    Pings every connected service and returns their status.
    Use this endpoint for uptime monitoring and deployment readiness checks.
    """
    results: Dict[str, str] = {}

    # Qdrant
    try:
        vs = get_vector_store()
        vs.get_collection_info()
        results["qdrant"] = "ok"
    except Exception as exc:
        results["qdrant"] = f"error: {exc}"

    # Groq
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        client.models.list()
        results["groq"] = "ok"
    except Exception as exc:
        results["groq"] = f"error: {exc}"

    # Google Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
        list(genai.list_models())
        results["gemini"] = "ok"
    except Exception as exc:
        results["gemini"] = f"error: {exc}"

    # OpenAI
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        openai.models.list()
        results["openai"] = "ok"
    except Exception as exc:
        results["openai"] = f"error: {exc}"

    # Anthropic
    try:
        import anthropic
        c = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        c.models.list()
        results["anthropic"] = "ok"
    except Exception as exc:
        results["anthropic"] = f"error: {exc}"

    overall = "healthy" if results["qdrant"] == "ok" and results["groq"] == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=results["qdrant"],
        groq=results["groq"],
        gemini=results["gemini"],
        openai=results["openai"],
        anthropic=results["anthropic"],
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ══════════════════════════════════════════════════════════════════════
# Root
# ══════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":    "Smart RAG Universal API",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/api/health",
    }
