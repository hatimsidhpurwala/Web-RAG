"""
Central configuration for the Smart Web Scraper with Hybrid RAG System.
All tunable parameters are gathered here for easy management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MD_FILES_DIR = DATA_DIR / "md_files"
QDRANT_LOCAL_DIR = PROJECT_ROOT / "qdrant_local"

# Load environment variables from config/.env
load_dotenv(CONFIG_DIR / ".env")

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER", "")

# ---------------------------------------------------------------------------
# LLM Configuration (Groq)
# ---------------------------------------------------------------------------
# Primary model — best quality, 100K tokens/day on free tier.
# The system will automatically fall back if this hits its rate limit.
LLM_MODEL: str = "llama-3.3-70b-versatile"

# Ordered list of fallback models tried if LLM_MODEL is rate-limited or decommissioned.
# Each has its own separate daily quota bucket.
MODEL_FALLBACK_CHAIN: list = [
    "llama-3.3-70b-versatile",          # Primary: 70B, best quality
    "qwen/qwen3-32b",                    # Fallback 1: 32B Qwen3 (preview)
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Fallback 2: Llama 4 Scout
    "llama-3.1-8b-instant",             # Last resort: 8B, minimal quality
]
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 2000

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ---------------------------------------------------------------------------
# Document Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 800          # tokens per chunk (increased for better context)
CHUNK_OVERLAP: int = 100       # overlapping tokens to preserve sentence continuity

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "scraped_pages"
TOP_K_RESULTS: int = 12          # ↑ more chunks → richer context
TOP_K_PER_QUERY: int = 5         # ↑ broader per-query sweep
SIMILARITY_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------
INTENT_CONFIDENCE_THRESHOLD: float = 0.7

# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------
WEB_SEARCH_CONFIDENCE_THRESHOLD: float = 0.60   # ↓ trigger research more readily
WEB_SEARCH_MAX_RESULTS: int = 5                  # ↑ scrape more pages per search
DEEP_RESEARCH_NUM_QUERIES: int = 5               # ↑ broader deep research
SCRAPE_DELAY_SECONDS: float = 1.0   # delay between domain requests

# ---------------------------------------------------------------------------
# Whisper (Voice)
# ---------------------------------------------------------------------------
WHISPER_MODEL: str = "whisper-large-v3"

# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------
TESSERACT_CMD: str = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
