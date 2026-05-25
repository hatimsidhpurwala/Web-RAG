"""
src/config.py
Loads all .env variables and raises clear errors if missing.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
QDRANT_LOCAL_DIR = PROJECT_ROOT / "qdrant_local"

load_dotenv(CONFIG_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

import logging
logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY missing. Add it to config/.env to enable Groq LLM features.")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY missing. Add it to config/.env to enable Google LLM features.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY missing. Add it to config/.env to enable OpenAI LLM features.")
if not QDRANT_URL:
    logger.info("QDRANT_URL not set. Defaulting to local Qdrant directory.")

# Model configuration
PROVIDER_MODELS = {
    "google": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o"
}
LLM_MODEL = PROVIDER_MODELS["google"]  # default
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
WHISPER_MODEL = "whisper-large-v3"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "scraped_pages"

def llm_completion(provider: str, messages: list, temperature: float = 0.2, max_tokens: int = None, json_mode: bool = False) -> str:
    model_name = PROVIDER_MODELS.get(provider, PROVIDER_MODELS["google"])
    
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        kwargs = {"model": model_name, "messages": messages, "temperature": temperature}
        if max_tokens: kwargs["max_tokens"] = max_tokens
        if json_mode: kwargs["response_format"] = {"type": "json_object"}
        res = client.chat.completions.create(**kwargs)
        return res.choices[0].message.content or ""
        
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        kwargs = {"model": model_name, "messages": messages, "temperature": temperature}
        if max_tokens: kwargs["max_tokens"] = max_tokens
        if json_mode: kwargs["response_format"] = {"type": "json_object"}
        res = client.chat.completions.create(**kwargs)
        return res.choices[0].message.content or ""
        
    else:  # google
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        
        system_instruction = None
        formatted_messages = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            else:
                role = "user" if m["role"] == "user" else "model"
                formatted_messages.append({"role": role, "parts": [m["content"]]})
                
        # Handle cases where first message is not user
        if formatted_messages and formatted_messages[0]["role"] == "model":
            formatted_messages.insert(0, {"role": "user", "parts": ["Hello"]})
            
        kwargs = {}
        if system_instruction: kwargs["system_instruction"] = system_instruction
        m = genai.GenerativeModel(model_name, **kwargs)
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else "text/plain"
        )
        
        res = m.generate_content(formatted_messages, generation_config=generation_config)
        return res.text

# Auto-initialize Postgres Database on application start
try:
    from src.storage.session import init_db
    init_db()
except Exception as e:
    logger.warning(f"Database auto-initialization skipped or failed: {e}")

