"""
src/api/server.py
All FastAPI routes
"""

import pathlib
import logging
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.agent.pipeline import RAGAgent
from src.ingestion.parser import parse, transcribe_audio, extract_from_image
from src.ingestion.chunker import process
from src.storage.vector_store import VectorStore, embed_chunks
from src.api.webhook import router as webhook_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Universal Scraper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = RAGAgent()
vector_store = VectorStore()

# Include the webhook router
app.include_router(webhook_router)


from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    session_id: str
    uploaded_docs: bool = False
    provider: str = "google"


@app.post("/api/ask")
async def ask(body: AskRequest):
    answer = agent.ask(body.question, body.session_id, body.uploaded_docs, provider=body.provider)
    return {"answer": answer}


def index_in_background(raw: bytes, filename: str, session_id: str):
    """Asynchronously parses, chunks, embeds, and stores the file."""
    try:
        text = parse(raw, filename)
        chunks_str = process(text)
        if chunks_str:
            site_name = f"{session_id}:doc:{filename}"
            chunks = [{"text": c, "source_url": filename} for c in chunks_str]
            chunks = embed_chunks(chunks)
            vector_store.store_chunks_for_site(chunks, site_name)
            logger.info(f"Background indexing completed for file: {filename} ({len(chunks_str)} chunks)")
    except Exception as e:
        logger.error(f"Error during background indexing of {filename}: {e}")

@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    session_id: str = Form(...)
):
    raw = await file.read()
    filename = file.filename or "file"
    
    # Quick early validation of file extension
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in [".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".txt", ".png", ".jpg", ".jpeg", ".wav", ".mp3"]:
        raise HTTPException(status_code=415, detail=f"Unsupported file format: {ext}")
        
    # Queue the heavy indexing task to run in the background immediately
    background_tasks.add_task(index_in_background, raw, filename, session_id)
    
    return {
        "status": "processing", 
        "message": f"Successfully queued '{filename}' for background parsing and indexing."
    }


@app.post("/api/voice")
async def voice(
    audio: UploadFile = File(...), 
    session_id: str = Form(...),
    provider: str = Form("google")
):
    raw = await audio.read()
    ext = pathlib.Path(audio.filename or "audio.wav").suffix.lower()
    text = transcribe_audio(raw, ext)
    answer = agent.ask(text, session_id, uploaded_docs=False, provider=provider)
    return {"transcription": text, "answer": answer}


@app.post("/api/ocr")
async def ocr(image: UploadFile = File(...), session_id: str = Form(...)):
    raw = await image.read()
    text = extract_from_image(raw)
    answer = agent.ask(text, session_id, uploaded_docs=False)
    return {"extracted_text": text, "answer": answer}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
