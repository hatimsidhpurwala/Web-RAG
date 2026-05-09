# 🧠 Smart Web Scraper with Hybrid RAG System

An intelligent AI assistant that scrapes websites, processes multiple input types (text, voice, images, PDFs), builds a knowledge base, and answers questions with **automatic web research** capabilities.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Hybrid RAG Pipeline** | LangGraph-based agent that classifies intent, retrieves from vector DB, and auto-searches the web when confidence is low |
| 🎙️ **Voice Input** | Audio transcription via Groq Whisper |
| 🖼️ **Image OCR** | Text extraction from images via Tesseract |
| 📄 **Document Profiling** | **NEW:** Smart hashing, AI-driven summaries, and deduplication of uploaded PDFs via Metadata Registry |
| 🌐 **Web Scraping** | Auto-detects URLs, scrapes, and indexes content |
| 🔬 **Deep Research** | `/research <topic>` command for comprehensive multi-source analysis |
| 💬 **WhatsApp Integration** | Twilio-powered webhook for WhatsApp conversations |
| 📊 **Confidence Scoring** | Visual confidence bars and source attribution |

---

## 🏗️ Architecture

```
       [ USER ]
          │
          │ 1. Uploads PDFs
          │ 2. Asks a Question
          ▼
+-----------------------------------+
|        STREAMLIT WEB UI           |
+-----------------------------------+
          │
          │ (When Uploading)
          ▼
+-----------------------------------+
|     DOCUMENT PROFILER AGENT       |
|  - Extracts text separately       |
|  - Asks AI to summarize each PDF  |
+-----------------------------------+
          │                   │
  (Saves  │                   │ (Saves Chunked
 Profiles)│                   │  Text with Tags)
          ▼                   ▼
+-------------------+ +-------------------+
| METADATA REGISTRY | | QDRANT VECTOR DB  |
+-------------------+ +-------------------+
          ▲                   ▲
  (Checks │                   │ (Searches ONLY
   Map)   │                   │  Target PDF)
          │                   │
+-----------------------------------+
|         QUERY ROUTER AGENT        |
|  - Gets User Question             |
|  - Consults Metadata Registry     |
|  - Pulls context from Vector DB   |
|  - Synthesizes Answer via AI      |
+-----------------------------------+
          │ (Auto-triggers Web Fallback if needed)
          ▼
       [ USER ]
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd universal-web-scraper
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure

```bash
copy config\.env.example config\.env
# Edit config/.env and add your GROQ_API_KEY
```

### 3. (Optional) Index existing data

```bash
python src/utils/indexer.py
```

### 4. Run the Streamlit UI

```bash
streamlit run src/api/streamlit_app.py
```

### 5. Run the WhatsApp webhook (separate terminal)

```bash
uvicorn src.api.whatsapp_webhook:app --reload --port 8000
```

---

## 🔧 Configuration

All settings live in `config/settings.py` and are overridable via `config/.env`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for LLM + Whisper |
| `QDRANT_URL` | *(local)* | Qdrant Cloud URL (leave blank for local) |
| `QDRANT_API_KEY` | *(local)* | Qdrant Cloud API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `CHUNK_SIZE` | `500` | Tokens per chunk |
| `TOP_K_RESULTS` | `8` | Number of retrieved chunks |

---

## 🐳 Docker

```bash
docker build -t smart-rag .
docker run -p 8000:8000 --env-file config/.env smart-rag
```

---

## 📁 Project Structure

```
universal-web-scraper/
├── src/
│   ├── agents/           # AI Agent Components
│   │   ├── models.py             # Pydantic structured outputs
│   │   ├── intent_classifier.py  # Intent classification
│   │   ├── query_generator.py    # Query optimisation
│   │   ├── retriever.py          # Vector DB retrieval
│   │   ├── response_generator.py # Answer generation
│   │   ├── direct_responder.py   # Non-retrieval responses
│   │   ├── web_searcher.py       # Web search & scraping
│   │   └── agent_graph.py        # LangGraph orchestrator
│   ├── core/             # Core Processing
│   │   ├── scraper.py            # HTML → Markdown
│   │   ├── cleaner.py            # Text normalisation
│   │   ├── chunker.py            # Document chunking
│   │   └── embedder.py           # Embedding generation
│   ├── database/         # Databases
│   │   ├── metadata_registry.py  # Stores document profiles
│   │   └── vector_store.py       # Qdrant operations
│   ├── api/              # User Interfaces
│   │   ├── streamlit_app.py      # ChatGPT-style UI
│   │   └── whatsapp_webhook.py   # WhatsApp webhook
│   └── utils/            # Utilities
│       ├── indexer.py            # Batch indexing
│       └── assets.py             # Constants
├── config/               # Configuration
│   ├── .env.example
│   └── settings.py
├── data/md_files/        # Markdown data files
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📝 Special Commands

| Command | Description |
|---|---|
| `/research <topic>` | Triggers deep web research (3 queries, 6 pages) |
| Paste a URL | Auto-scrapes and indexes the page |
| Upload a PDF | Auto-extracts text and indexes into the knowledge base |

---

## 🔒 Security

- API keys stored in `.env` (never committed)
- WhatsApp sessions are in-memory only
- Vector DB contains text chunks only (no PII)
- User-agent rotation for web scraping
- Rate limiting on external API calls

---

## 📄 License

MIT
