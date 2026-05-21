# 🧠 Universal Web Scraper & API with Hybrid RAG

A state-of-the-art, multi-modal AI architecture that ingests text, images, voice, PDFs, and scraped web content. This project provides a universal brain via a flexible REST API and a robust Streamlit UI, allowing deployment across WhatsApp, Telegram, Instagram, and web applications natively.

---

## ✨ Core Features

| Feature | Description |
|---|---|
| 🌍 **Universal Webhook API** | Exposes a single `/api/webhook/universal` endpoint that magically auto-detects and answers payloads from **Telegram, Instagram, Twilio (WhatsApp), Zapier, or Make.com**. |
| 🗄️ **Omnichannel Memory** | A centralized `session_manager` ensures that if a user starts a conversation on WhatsApp and finishes it on the web UI, the AI remembers perfectly. |
| 📄 **Universal File Parser** | `document_parser.py` seamlessly handles PDF, Word, Excel, CSV, PPT, and plain text uploads identically across both the API and Streamlit UI. |
| 🎙️ **Voice & 🖼️ Image** | Natively transcribes audio (Groq Whisper) and extracts text from images (Tesseract OCR). |
| 🤖 **Multi-Model Hot Swapping** | Dynamically hot-swap between Groq, Gemini, OpenAI, and Anthropic in real-time. |
| 🌐 **Autonomous Web Scraping** | The Agent recognizes knowledge gaps, searches the internet via DuckDuckGo, scrapes the HTML, embeds it into Qdrant, and returns factual answers. |

---

## 🏗️ Architectural Pipeline

```
       [ Any Platform: Web / Telegram / WhatsApp / REST ]
                               │
                               ▼
+---------------------------------------------------------------+
|                      UNIVERSAL FASTAPI                        |
|   /api/webhook/universal   |   /api/ask   |   /api/upload     |
+---------------------------------------------------------------+
                               │
            +------------------+------------------+
            ▼                                     ▼
+-------------------------+            +--------------------------+
|  document_parser.py     |            | session_manager.py       |
| (PDF, Excel, Word, OCR) |            | (Persistent JSON state)  |
+-------------------------+            +--------------------------+
            │                                     │
            +------------------+------------------+
                               ▼
+---------------------------------------------------------------+
|                 LANGGRAPH AGENT ORCHESTRATOR                  |
| 1. Intent Classification          4. Web Fallback             |
| 2. Multimodal Fusion              5. Synthesis & Fact Check   |
| 3. Qdrant Vector DB Retrieval     6. Confidence Scoring       |
+---------------------------------------------------------------+
```

---

## 🚀 Quick Start

### 1. Clone & Install Environment

```bash
git clone <repo-url>
cd universal-web-scraper
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment (`config/.env`)

Copy the example environment file:
```bash
copy config\.env.example config\.env
```

Ensure your `config/.env` contains your active keys:
```env
# Required for core LLM routing
GROQ_API_KEY=gsk_your_groq_api_key

# Required for Universal Models (Gemini routing)
GOOGLE_API_KEY=AIza_your_google_key

# Required for Twilio integration (Optional for universal)
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_WHATSAPP_NUMBER=+14155238886

# Required for Persistent Vector Storage (Cloud)
QDRANT_URL=https://your-qdrant-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key
```

### 3. Run the Omnichannel System

The easiest way to start both the Universal REST API (FastAPI) and the Streamlit UI simultaneously:

```bash
python run_all.py
```
- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI Backend:** `http://localhost:8000`
- **API Documentation:** `http://localhost:8000/docs`

---

## 🌐 How to Use the Universal Webhook

If you are connecting this AI to **Telegram, Instagram, Twilio**, or standard webhooks, point your app to:
`POST http://<your-server-ip>:8000/api/webhook/universal`

The webhook automatically parses:
- `application/x-www-form-urlencoded` (Twilio WhatsApp)
- `application/json` containing Telegram Webhook format (`message.chat.id`)
- `application/json` containing Facebook/Instagram format (`entry.messaging.sender.id`)
- Standard `application/json` (`{"session_id": "123", "message": "hello"}`)

You no longer need specific endpoint mapping for standard integrations.

---

## 📁 Modular Project Structure

```
universal-web-scraper/
├── src/
│   ├── core/             # Data processing & extraction
│   │   ├── document_parser.py    # Universal doc extraction (PDF, Excel, Word, etc)
│   │   ├── scraper.py            # HTML → Markdown
│   │   ├── cleaner.py            # Text normalisation
│   │   ├── chunker.py            # Document chunking
│   │   └── embedder.py           # Embedding generation
│   ├── database/         # Data persistence & state management
│   │   ├── session_manager.py    # Centralized persistent JSON chat memory
│   │   ├── metadata_registry.py  # Stores document profiles
│   │   └── vector_store.py       # Qdrant operations
│   ├── api/              # Ingress Endpoints
│   │   ├── rest_api.py           # Universal FastAPI backend & Webhooks
│   │   ├── streamlit_app.py      # Main Chat UI
│   │   └── whatsapp_webhook.py   # Legacy direct twilio adapter
│   ├── ui/               # UI Rendering Modules
│   │   ├── renderers.py          # Visual component rendering 
│   │   └── styles.py             # Global CSS abstraction
│   ├── agents/           # LangGraph Agent logic
│   │   ├── agent_graph.py        # Central Orchestrator
│   │   └── ...                   # Sub-agent implementations
│   └── utils/
├── config/               # Settings & API keys
├── data/                 # JSON memory & markdown storage
├── requirements.txt      # Core Dependencies
├── run_all.py            # Start script
└── README.md
```

---

## 🔒 Security & Privacy

- **Memory Isolation:** The `session_manager` rigorously namespaces memory by `session_id` to ensure users never access another user's chat history.
- **Vector Isolation:** Qdrant payloads explicitly attach a unique `site_name` tag prefixing the user's `session_id`, sandboxing knowledge bases.
- **Stateless Agent:** The underlying LangGraph agent is ephemeral and only reconstructs history dynamically on invocation. 

---

## 📄 License
MIT
