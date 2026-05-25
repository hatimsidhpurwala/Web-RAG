# 🧠 Production-Grade Web-RAG platform (SQLite, PostgreSQL, Qdrant Cloud, Async Workers, Hybrid Search)

A state-of-the-art, highly optimized RAG (Retrieval-Augmented Generation) application designed for enterprise deployment. This system combines multi-user conversational memory via a local/remote **PostgreSQL database**, serverless cloud **Qdrant Vector storage**, asynchronous **background task worker queues**, and high-fidelity local **Jaccard Token-Overlap Hybrid Reranking**—engineered cleanly into exactly **15 python files**.

---

## 🛠️ Technology Stack
* **Web Interface:** Streamlit (vibrant modern styling with glassmorphic CSS overlays)
* **API Backend:** FastAPI (Asynchronous REST API + Uvicorn)
* **Primary Database (Chat Log & Session Memory):** PostgreSQL (16+)
* **Vector Database:** Qdrant Cloud (Stateless cloud clusters)
* **Embedding Model:** `all-MiniLM-L6-v2` (Local Sentence-Transformers, 384-dimensional dense vectors)
* **RAG LLM Engine:** Gemini 2.5 Flash (Default Google model), Groq (Llama 3.3), OpenAI (GPT-4o)
* **Multimodal Extraction:**
  * **Document Parsing:** Python PDFMiner, DOCX parser, Pandas Excel/CSV reader, PPTX analyzer.
  * **Voice Transcription:** Groq Whisper Large v3
  * **OCR Engine:** Tesseract OCR (v5+)

---

## 🏗️ Project Directory (15-File Debug Guide)
For swift debugging and code navigation, here is the absolute one-liner responsibility of each file:

```text
src/
├── ingestion/
│   ├── __init__.py      # Marks directory as a python module.
│   ├── parser.py        # COMBINED input parsing: PDF, DOCX, CSV/Excel, PPTX, Image OCR, Audio Whisper.
│   └── chunker.py       # Cleans document artifacts and splits text into clean 500-word blocks.
├── storage/
│   ├── __init__.py      # Marks directory as a python module.
│   ├── session.py       # Manages PostgreSQL connections and reads/writes chat logs & doc summaries.
│   └── vector_store.py  # Coordinates cloud/local Qdrant collections, upserts, and semantic search.
├── agent/
│   ├── __init__.py      # Marks directory as a python module.
│   ├── pipeline.py      # Main entry point (RAGAgent); checks if intent is a greeting, RAG, or Web Search.
│   ├── retriever.py     # Rewrites user queries contextually and executes Jaccard Hybrid Keyword Re-ranking.
│   ├── web_search.py    # Backs up empty vector stores with live DuckDuckGo scraping and vector embedding.
│   └── responder.py     # Generates cited AI answers with exact source URLs based on retrieved context.
├── api/
│   ├── __init__.py      # Marks directory as a python module.
│   ├── server.py        # Exposes all FastAPI endpoints (/ask, /upload, /voice, /ocr) with BackgroundTasks.
│   ├── webhook.py       # The universal endpoint (/api/webhook/universal) that connects to WhatsApp/Telegram.
│   └── ui.py            # Streamlit Chat interface, complete with custom CSS styling and audio inputs.
└── config.py            # Centerpiece environment loader (checks config/.env and loads global variables).

run.py                   # Master root launcher (boots Uvicorn server on port 8000 and Streamlit on port 8501).
```

---

## ⚙️ Pipeline Flow & Architecture
This diagram outlines the sequential flow of a user message and documents through the advanced, stateless pipeline:

```mermaid
graph TD
    User([Public User]) -->|Upload File| UI[ui.py: Streamlit]
    User -->|Message / Voice| UI
    
    %% API Routing
    UI -->|Async HTTP POST| API[server.py: FastAPI]
    
    subgraph Background_Processing ["Background Processing (FastAPI BackgroundTasks Queue)"]
        API -->|Task Queued: 202 Accepted| BG[server.py: index_in_background]
        BG -->|Parse| P[ingestion/parser.py]
        P -->|Chunk| C[ingestion/chunker.py]
        C -->|Embed| VS[storage/vector_store.py]
        VS -->|Upsert Chunks| QC[(Qdrant Cloud)]
    end
    
    %% RAG Pipeline Flow
    API -->|Ask Query| PIPE[agent/pipeline.py]
    PIPE -->|Read State| DB[(Local/VM PostgreSQL)]
    DB -->|Fetch History| PIPE
    
    PIPE -->|Rewrite Query| RET[agent/retriever.py]
    RET -->|Fetch Candidates| QC
    QC -->|Top 10 Dense Vectors| RET
    RET -->|Jaccard Hybrid Keyword Rerank| HBR[retriever.py: Re-sorted Top-5 Chunks]
    
    HBR --> PIPE
    PIPE -->|Generate cited answer| RESP[agent/responder.py]
    RESP -->|Save Exchange| DB
    RESP -->|Return Answer| API
    API -->|Send JSON response| User
```

---

## 🚀 How to Setup and Run Locally

### 1. Prerequisite Installations
Before running, ensure your machine has the following third-party dependencies installed:
* **Tesseract OCR (for Image parsing):** Download the [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki) and ensure the executable path matches your `config/.env` or `src/config.py` (Default: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
* **PostgreSQL:** Follow the detailed setup guide in the next section.

### 2. Setup the Repository
Clone, create your virtual environment, and install dependencies:
```powershell
# 1. Clone the project
git clone https://github.com/hatimsidhpurwala/Web-RAG.git
cd Web-RAG

# 2. Create virtual environment
python -m venv venv

# 3. Activate the environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install all Python packages
pip install -r requirements.txt
```

### 3. Setup the Configuration Environment
Create your config file in the `config/` directory:
- Copy `config/.env.example` and name it `config/.env`.
- Fill in your API keys:
  ```text
  GROQ_API_KEY=gsk_your_groq_key
  GOOGLE_API_KEY=AIzaSy_your_gemini_key
  OPENAI_API_KEY=sk-proj-your_openai_key
  
  # PostgreSQL connection URL (See Postgres setup section for details)
  DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/web_rag
  
  # For local Qdrant, leave empty. For Qdrant Cloud:
  QDRANT_URL=https://your-cloud-cluster.aws.cloud.qdrant.io:6333
  QDRANT_API_KEY=your_qdrant_cloud_api_key
  ```

### 4. How to Start Everything
You do not need to open multiple terminals to launch the backend and frontend. The master script `run.py` automatically spins up both Uvicorn and Streamlit in parallel using Python subprocesses!
```powershell
python run.py
```
* **API server is running at:** [http://localhost:8000](http://localhost:8000)
* **Streamlit Chat UI is running at:** [http://localhost:8501](http://localhost:8501)

---

## 🐘 Detailed PostgreSQL Setup & Inspection Guide

### 1. Local / VM Installation
* **Windows (Manual Installer):** Download and run the stable Windows installer from the [PostgreSQL website](https://www.postgresql.org/download/windows/). Keep the default port as `5432` and set your password (e.g. `hatsid`).
* **Linux VM (Ubuntu):** Run these commands inside your VM terminal:
  ```bash
  sudo apt update
  sudo apt install postgresql postgresql-contrib -y
  sudo systemctl start postgresql
  ```

### 2. Creating the Database
Before running the application, PostgreSQL requires a dedicated database. Open your terminal and run the following command (replace `postgres` with your username and `5432` with your port if modified):
```powershell
# Set PGPASSWORD so you aren't prompted interactively
$env:PGPASSWORD="YOUR_PASSWORD_HERE"

# Create the database
& "C:\Program Files\PostgreSQL\16\bin\createdb.exe" -U postgres -p 5432 web_rag
```
*(On Linux VMs, simply run: `sudo -u postgres createdb web_rag`)*

### 3. Inspecting Tables & Verifying Chat History
You can inspect what tables were created and verify the logged chat history using the PostgreSQL command-line tool `psql` or the **pgAdmin 4** GUI:

#### Option A: Command Line (`psql`)
Open your terminal and connect to your database:
```powershell
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -d web_rag
```
Once connected to the postgres prompt (`web_rag=#`), run these diagnostics:

* **Show all tables:**
  ```sql
  \dt
  ```
  *Output will show:*
  ```text
                 List of relations
   Schema |     Name     | Type  |  Owner
  --------+--------------+-------+----------
   public | chat_history | table | postgres
   public | doc_summaries| table | postgres
  ```

* **Inspect Chat Logs & Multi-User memory:**
  To see if your phone numbers (UIDs) are logging conversations correctly:
  ```sql
  SELECT id, session_id, role, LEFT(content, 40) AS content, created_at FROM chat_history ORDER BY created_at DESC;
  ```

* **Inspect Parsed Document Summaries:**
  ```sql
  SELECT * FROM doc_summaries;
  ```

* **Exit psql:**
  ```sql
  \q
  ```

#### Option B: pgAdmin 4 (GUI)
1. Open pgAdmin 4.
2. Under **Servers**, connect to `PostgreSQL 16`.
3. Expand **Databases** -> **web_rag** -> **Schemas** -> **public** -> **Tables**.
4. Right-click `chat_history` and select **View/Edit Data** -> **All Rows** to view your chat memory logs visually!

---

## 📡 Live API Webhook & Integration Guide (WhatsApp / Telegram)

The server contains a universal webhook handler designed to connect to automated messaging APIs seamlessly.

### 1. The Webhook Endpoint
The master endpoint for external application triggers is:
`POST http://<your-domain>/api/webhook/universal`

#### Request Payload Structure (JSON)
Incoming HTTP requests from Telegram, WhatsApp, or twilio webhooks must hit the endpoint with this payload:
```json
{
  "message": "User question here",
  "session_id": "+919876543210",
  "platform": "whatsapp"
}
```
* **`message`:** The text message sent by the user.
* **`session_id`:** The unique phone number or chat ID of the user. The backend automatically associates this with their Postgres chat history.
* **`platform`:** (Optional) Source platform (e.g. `"whatsapp"`, `"telegram"`, `"web"`).

### 2. How to Connect Webhooks Live (WhatsApp & Telegram)
Because local web servers (`localhost`) cannot be reached by Meta (WhatsApp) or Telegram servers, you must expose your local FastAPI port (`8000`) to the public web during testing using **ngrok**:

1. **Install and run ngrok:**
   ```bash
   ngrok http 8000
   ```
2. **Copy the Secure URL:**
   Copy the generated HTTPS URL (e.g., `https://a1b2-34-56-78.ngrok-free.app`).
3. **Register Webhook:**
   * **WhatsApp (Meta Developer Console):** Paste `https://a1b2-34-56-78.ngrok-free.app/api/webhook/universal` into the Webhook Callback URL section.
   * **Telegram (via API hit):** Register your webhook address using a simple GET query:
     `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook?url=https://a1b2-34-56-78.ngrok-free.app/api/webhook/universal`
4. **Instant Flow:** When a user sends a message on WhatsApp/Telegram, Meta/Telegram sends a POST request containing the user's phone number as the `session_id`. The backend loads the user's Postgres history, executes Jaccard Hybrid re-ranking, and responds instantly!
