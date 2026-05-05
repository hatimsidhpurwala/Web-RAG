"""
Enhanced Streamlit ChatGPT-style UI – integrated with all advanced
features: multi-agent collaboration, fact verification, reasoning
traces, sentiment awareness, conversation branching, multilingual
support, memory & learning, and self-improvement analytics.
"""

from __future__ import annotations

import io
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st


# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GROQ_API_KEY, TESSERACT_CMD, WHISPER_MODEL
from src.agents.agent_graph import RAGAgent
from src.agents.multimodal_fusion import MultimodalFusion
from src.core.chunker import chunk_markdown
from src.core.cleaner import deduplicate_chunks, normalize_text
from src.core.embedder import embed_chunks
from src.core.scraper import scrape_website
from src.database.vector_store import VectorStore
from src.utils.assets import (
    APP_DESCRIPTION,
    APP_NAME,
    CHAT_AVATAR_AI,
    CHAT_AVATAR_USER,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    URL_PATTERN,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Page config
# ======================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Custom CSS – premium dark UI
# ======================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc;
}

.stChatMessage {
    background: rgba(30, 27, 75, 0.45) !important;
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.3s ease;
}
.stChatMessage:hover {
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.08);
}

.stChatInputContainer {
    background: rgba(15, 12, 41, 0.8) !important;
    border-top: 1px solid rgba(99, 102, 241, 0.15);
    backdrop-filter: blur(16px);
}
.stChatInputContainer textarea {
    background: rgba(30, 27, 75, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    color: #e0e7ff !important;
}
.stChatInputContainer textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.25rem;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
}

.stFileUploader {
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    background: rgba(30, 27, 75, 0.3) !important;
}
.stFileUploader:hover {
    border-color: rgba(99, 102, 241, 0.6) !important;
}

.streamlit-expanderHeader {
    background: rgba(30, 27, 75, 0.5) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}

[data-testid="stMetric"] {
    background: rgba(30, 27, 75, 0.4);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    padding: 0.85rem 1rem;
}

.stAlert { border-radius: 12px !important; border: none !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }

.source-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 2px 4px 2px 0;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.feature-badge {
    display: inline-block;
    background: rgba(34, 197, 94, 0.12);
    color: #86efac;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    margin: 1px 3px 1px 0;
    border: 1px solid rgba(34, 197, 94, 0.2);
}

.confidence-bar {
    height: 6px;
    border-radius: 3px;
    margin-top: 6px;
    transition: width 0.6s ease;
}

.fact-verified { color: #22c55e; font-size: 0.8rem; }
.fact-unverified { color: #eab308; font-size: 0.8rem; }
.fact-contradicted { color: #ef4444; font-size: 0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================
# Session state
# ======================================================================

@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_agent(_vector_store):
    return RAGAgent(_vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()
if "agent" not in st.session_state:
    st.session_state.agent = get_agent(st.session_state.vector_store)
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None
if "active_doc_sites" not in st.session_state:
    # Tracks site_names of all documents uploaded in this session.
    # Passed to agent.ask() so retrieval can be restricted to pdf_ sources
    # when the user asks about an uploaded document.
    st.session_state.active_doc_sites: list[str] = []


# ======================================================================
# Multi-modal helpers
# ======================================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", f), model=WHISPER_MODEL,
            language="en", response_format="text",
        )
    Path(tmp_path).unlink(missing_ok=True)
    return transcription.strip() if isinstance(transcription, str) else str(transcription).strip()


def ocr_image(image_bytes: bytes) -> str:
    import cv2
    import numpy as np
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)
    return pytesseract.image_to_string(denoised, lang="eng").strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def detect_urls(text: str) -> List[str]:
    return re.findall(URL_PATTERN, text)


def index_text_content(text: str, source_name: str, vs: VectorStore) -> int:
    text = normalize_text(text)
    chunks = chunk_markdown(text, source_url=source_name)
    chunks = deduplicate_chunks(chunks)
    chunks = embed_chunks(chunks, show_progress=False)
    return vs.store_chunks_for_site(chunks, source_name)


def process_input(text: str, file=None, audio_bytes=None, force_reindex: bool = False) -> dict:
    vs = st.session_state.vector_store
    actions, scraped_sites = [], []
    processed_text = text or ""
    input_type = "text"
    ocr_text = ""
    pdf_text = ""

    if audio_bytes:
        with st.spinner("🎙️ Transcribing audio…"):
            transcription = transcribe_audio(audio_bytes)
        processed_text = transcription
        input_type = "voice"
        actions.append(f"🎙️ Transcribed: {len(transcription)} chars")

    if file is not None:
        ext = Path(file.name).suffix.lower()
        raw = file.read()

        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            with st.spinner("🖼️ Running OCR…"):
                ocr_text = ocr_image(raw)
            processed_text = ocr_text
            input_type = "image"
            actions.append(f"🖼️ OCR: {len(ocr_text)} chars")

        elif ext in SUPPORTED_PDF_EXTENSIONS:
            with st.spinner("📄 Extracting PDF…"):
                pdf_text = extract_pdf_text(raw)
            site_name = f"pdf_{Path(file.name).stem}"

            already_exists = vs.has_site(site_name)
            if force_reindex and already_exists:
                with st.spinner("🔄 Re-indexing with deep chunking…"):
                    vs.clear_site(site_name)
                    stored = index_text_content(pdf_text, site_name, vs)
                actions.append(f"📄 Deep re-indexed PDF → {stored} chunks")
            elif not already_exists:
                with st.spinner("📦 Indexing PDF…"):
                    stored = index_text_content(pdf_text, site_name, vs)
                actions.append(f"📄 Indexed PDF → {stored} chunks")
            else:
                actions.append(f"📄 PDF already indexed as '{site_name}'")

            processed_text = pdf_text[:500] + ("…" if len(pdf_text) > 500 else "")
            input_type = "pdf"
            scraped_sites.append(site_name)
            if site_name not in st.session_state.active_doc_sites:
                st.session_state.active_doc_sites.append(site_name)

    # URL detection
    urls = detect_urls(processed_text)
    if urls:
        for url in urls[:3]:
            with st.spinner(f"🌐 Scraping {url}…"):
                md = scrape_website(url)
            if md:
                from urllib.parse import urlparse
                from datetime import datetime
                domain = urlparse(url).netloc.replace("www.", "")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                site_name = f"web_{domain}_{ts}"
                stored = index_text_content(md, site_name, vs)
                actions.append(f"🌐 Scraped {url} → {stored} chunks")
                scraped_sites.append(site_name)

    # Multimodal fusion if multiple inputs
    if sum(bool(x) for x in [text, ocr_text, pdf_text, audio_bytes]) > 1:
        fusion = MultimodalFusion()
        fused = fusion.fuse_inputs(
            text_input=text or "",
            image_ocr=ocr_text,
            pdf_text=pdf_text,
            voice_transcription=processed_text if input_type == "voice" else "",
        )
        if fused.get("unified_query"):
            processed_text = fused["unified_query"]
            actions.append("🔗 Multimodal fusion applied")

    return {
        "processed_text": processed_text,
        "type": input_type,
        "actions_taken": actions,
        "scraped_sites": scraped_sites,
    }


# ======================================================================
# Render helpers
# ======================================================================

def render_confidence(conf: float):
    color = "#22c55e" if conf >= 0.8 else "#eab308" if conf >= 0.6 else "#ef4444"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='font-size:0.8rem;color:{color};'>Confidence: {conf:.0%}</span>"
        f"<div style='flex:1;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;'>"
        f"<div class='confidence-bar' style='width:{conf*100:.0f}%;background:{color};'></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_sources(sources: List[str]):
    if sources:
        html = " ".join(f"<span class='source-badge'>{s}</span>" for s in sources[:5])
        st.markdown(f"**Sources:** {html}", unsafe_allow_html=True)


def render_features(features: List[str]):
    if features:
        html = " ".join(f"<span class='feature-badge'>⚡ {f}</span>" for f in features)
        st.markdown(html, unsafe_allow_html=True)


def render_fact_check(report: dict):
    if not report:
        return
    reliability = report.get("overall_reliability", 0)
    summary = report.get("verification_summary", "")
    contradictions = report.get("contradictions", [])

    color_class = (
        "fact-verified" if reliability >= 0.8
        else "fact-unverified" if reliability >= 0.5
        else "fact-contradicted"
    )
    icon = "✅" if reliability >= 0.8 else "⚠️" if reliability >= 0.5 else "❌"

    st.markdown(
        f"<span class='{color_class}'>{icon} Fact Check: {summary}</span>",
        unsafe_allow_html=True,
    )
    if contradictions:
        with st.expander("⚠️ Contradictions found"):
            for c in contradictions:
                st.warning(c)


def render_meta(meta: dict):
    """Render all metadata for a message."""
    render_sources(meta.get("sources", []))

    if meta.get("confidence") is not None:
        render_confidence(meta["confidence"])

    if meta.get("fact_check_report"):
        render_fact_check(meta["fact_check_report"])

    if meta.get("web_search"):
        st.info("🌐 Web research was performed to improve this answer.")

    if meta.get("response_time_ms"):
        st.caption(f"⏱️ {meta['response_time_ms']}ms")

    render_features(meta.get("features", []))

    if meta.get("follow_ups"):
        with st.expander("💡 Suggested follow-ups"):
            for q in meta["follow_ups"]:
                st.markdown(f"• {q}")

    if meta.get("conversation_state"):
        cs = meta["conversation_state"]
        if cs.get("state") and cs["state"] != "new_topic":
            st.caption(f"🔀 Conversation: {cs['state']} • Topic: {cs.get('current_topic', '')[:50]}")


# ======================================================================
# Sidebar
# ======================================================================

with st.sidebar:
    st.markdown("# 🧠 Smart RAG")
    st.markdown("*An intelligent assistant that automatically researches, verifies facts, and learns from every conversation.*")
    st.divider()

    # File upload — multiple files supported
    st.markdown("### 📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Drop PDFs, images, or documents (multiple allowed)",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp", "docx"],
        key="file_upload",
        accept_multiple_files=True,
    )


    # Track which file names we've already processed this session
    # so we only re-index when a NEW file is added, not on every rerun
    if "_processed_file_names" not in st.session_state:
        st.session_state._processed_file_names: set = set()

    if uploaded_files:
        for uf in uploaded_files:
            file_key = f"{uf.name}_{uf.size}"
            is_new   = file_key not in st.session_state._processed_file_names
            result   = process_input("", file=uf, force_reindex=is_new)
            for action in result["actions_taken"]:
                if is_new:
                    st.success(action)
                else:
                    st.caption(f"✓ {action}")
            if is_new:
                st.session_state._processed_file_names.add(file_key)

    # ── Active Documents panel ──────────────────────────────────
    active_docs = st.session_state.get("active_doc_sites", [])
    if active_docs:
        st.markdown("**📂 Active Documents this session:**")
        for doc in active_docs:
            # Show just the filename part (strip pdf_ prefix)
            label = doc.replace("pdf_", "").replace("img_", "")
            st.markdown(f"&nbsp;&nbsp;📄 `{label}`", unsafe_allow_html=True)
        if st.button("🗑️ Clear active documents", use_container_width=True):
            st.session_state.active_doc_sites = []
            st.session_state._processed_file_names = set()
            st.rerun()

    st.divider()

    # ── Voice Input ────────────────────────────────────────────
    st.markdown("### 🎙️ Voice Input")

    voice_tab1, voice_tab2 = st.tabs(["🎙️ Speak Now", "📂 Upload File"])

    with voice_tab1:
        # Browser-native Web Speech API – no pip package, no backend
        import streamlit.components.v1 as components
        components.html(
            """
<style>
  body { margin:0; background:transparent; font-family:'Inter',sans-serif; }
  #wrap {
    display:flex; flex-direction:column; align-items:center;
    gap:10px; padding:12px 0;
  }
  #micBtn {
    width:60px; height:60px; border-radius:50%; border:none;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:white; font-size:26px; cursor:pointer;
    box-shadow:0 4px 18px rgba(99,102,241,0.45);
    transition:all .25s ease; outline:none;
  }
  #micBtn:hover { transform:scale(1.08); box-shadow:0 6px 24px rgba(99,102,241,0.6); }
  #micBtn.recording {
    background:linear-gradient(135deg,#ef4444,#dc2626);
    animation:pulse 1s infinite;
    box-shadow:0 0 0 0 rgba(239,68,68,.7);
  }
  @keyframes pulse {
    0%   { box-shadow:0 0 0 0 rgba(239,68,68,.7); }
    70%  { box-shadow:0 0 0 12px rgba(239,68,68,0); }
    100% { box-shadow:0 0 0 0 rgba(239,68,68,0); }
  }
  #status {
    font-size:12px; color:#a5b4fc; text-align:center;
    min-height:18px; max-width:180px; word-break:break-word;
  }
  #transcript {
    font-size:11px; color:#6ee7b7; text-align:center;
    max-width:180px; min-height:14px; font-style:italic;
  }
  #supportWarn { font-size:11px; color:#f87171; text-align:center; }
</style>
<div id="wrap">
  <button id="micBtn" title="Click to speak">🎙️</button>
  <div id="status">Click mic to speak</div>
  <div id="transcript"></div>
  <div id="supportWarn"></div>
</div>
<script>
  const btn      = document.getElementById('micBtn');
  const statusEl = document.getElementById('status');
  const txEl     = document.getElementById('transcript');
  const warnEl   = document.getElementById('supportWarn');

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    warnEl.textContent = '⚠️ Your browser does not support Speech Recognition. Try Chrome or Edge.';
    btn.disabled = true;
    btn.style.opacity = '0.4';
  }

  let recognition = null;
  let listening   = false;

  btn.addEventListener('click', () => {
    if (!SR) return;
    if (listening) { recognition.stop(); return; }
    startListening();
  });

  function startListening() {
    recognition = new SR();
    recognition.lang = 'en-US';          // will auto-detect in many browsers
    recognition.continuous      = false;
    recognition.interimResults  = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      listening = true;
      btn.classList.add('recording');
      btn.textContent  = '🔴';
      statusEl.textContent  = 'Listening…';
      txEl.textContent = '';
    };

    recognition.onresult = (e) => {
      const interim = Array.from(e.results)
        .map(r => r[0].transcript).join('');
      txEl.textContent = interim;
    };

    recognition.onerror = (e) => {
      statusEl.textContent = 'Error: ' + e.error;
      stopUI();
    };

    recognition.onend = () => {
      const final = txEl.textContent.trim();
      stopUI();
      if (final) {
        statusEl.textContent = '✓ Got it – sending…';
        submitToChat(final);
      } else {
        statusEl.textContent = 'Nothing heard – try again';
      }
    };

    recognition.start();
  }

  function stopUI() {
    listening = false;
    btn.classList.remove('recording');
    btn.textContent = '🎙️';
  }

  function submitToChat(text) {
    // Walk up to the top-level Streamlit frame and inject text
    try {
      const doc = window.parent.document;
      const textarea = doc.querySelector(
        '[data-testid="stChatInput"] textarea, '
        + '.stChatInput textarea'
      );
      if (!textarea) {
        statusEl.textContent = 'Could not find chat input. Try typing manually.';
        txEl.textContent = text;
        return;
      }
      // React-compatible value injection
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      ).set;
      setter.call(textarea, text);
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
      textarea.dispatchEvent(new Event('change', { bubbles: true }));

      // Give React time to sync, then click send
      setTimeout(() => {
        const send = doc.querySelector(
          '[data-testid="stChatInputSubmitButton"], '
          + 'button[kind="primaryFormSubmit"]'
        );
        if (send) { send.click(); }
        txEl.textContent = '';
        statusEl.textContent = 'Click mic to speak';
      }, 400);
    } catch(err) {
      statusEl.textContent = 'Could not auto-send: ' + err.message;
      txEl.textContent = text;
    }
  }
</script>
""",
            height=140,
            scrolling=False,
        )

    with voice_tab2:
        audio_file = st.file_uploader(
            "Upload WAV, MP3, M4A, OGG, or WEBM – transcribed by Groq Whisper",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
            key="audio_upload",
        )
        if audio_file:
            st.caption(f"📂 {audio_file.name} • {audio_file.size // 1000}KB — will transcribe on send")

    st.divider()

    # Knowledge base
    st.markdown("### 📊 Knowledge Base")
    try:
        info = st.session_state.vector_store.get_collection_info()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vectors", info.get("vectors_count", 0))
        with col2:
            st.metric("Points", info.get("points_count", 0))
    except Exception:
        st.info("KB not initialised.")

    try:
        indexed = st.session_state.vector_store.list_indexed_sites()
    except Exception:
        indexed = []
    if indexed:
        st.markdown("**Indexed Sources:**")
        for site in indexed[-10:]:
            st.markdown(f"<span class='source-badge'>{site}</span>", unsafe_allow_html=True)

    st.divider()

    # Memory & Learning stats (auto-loaded, no command needed)
    st.markdown("### 🧠 Memory & Learning")
    try:
        mem = st.session_state.agent.memory.get_stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Corrections", mem.get("corrections", 0),
                      help="Answers improved from your feedback")
        with c2:
            st.metric("FAQ", mem.get("faq_entries", 0),
                      help="Unique questions asked")

        top_q = st.session_state.agent.memory.get_top_queries(3)
        if top_q:
            st.markdown("**Top Questions:**")
            for tq in top_q:
                st.caption(f"({tq['count']}×) {tq['query'][:45]}…" if len(tq['query']) > 45 else f"({tq['count']}×) {tq['query']}")
    except Exception:
        pass

    st.divider()

    # Live Performance panel (replaces /improve command)
    st.markdown("### 📊 Performance")
    try:
        perf = st.session_state.agent.improvement_engine.analyze_performance()
        summary = perf.get("summary", {})
        if summary.get("total_interactions", 0) > 0:
            c1, c2 = st.columns(2)
            with c1:
                conf = summary.get("avg_confidence", 0)
                color = "normal" if conf >= 0.8 else "inverse"
                st.metric("Avg Confidence", f"{conf:.0%}")
            with c2:
                st.metric("Interactions", summary.get("total_interactions", 0))
            ws_rate = summary.get("web_search_rate", 0)
            st.caption(f"🌐 Auto-research triggered in {ws_rate:.0%} of queries")

            improvements = perf.get("improvements", [])
            if improvements:
                with st.expander("💡 System Insights", expanded=False):
                    for imp in improvements[:3]:
                        icon = "🔴" if imp["priority"] == "high" else "🟡"
                        st.caption(f"{icon} {imp['description'][:80]}")
        else:
            st.caption("Stats will appear after first conversation.")
    except Exception:
        st.caption("Stats will appear after first conversation.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ======================================================================
# Main chat area
# ======================================================================

st.markdown(
    "<h1 style='text-align:center; color:#a5b4fc; margin-bottom:0;'>🧠 Smart Web Scraper</h1>"
    "<p style='text-align:center; color:#6366f1; margin-top:4px;'>"
    "Hybrid RAG Assistant &bull; Auto Deep Research &bull; Fact-Verified &bull; Memory-Enabled</p>",
    unsafe_allow_html=True,
)

# Display history
for msg in st.session_state.messages:
    avatar = CHAT_AVATAR_USER if msg["role"] == "user" else CHAT_AVATAR_AI
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)
        meta = msg.get("meta", {})
        if meta:
            render_meta(meta)

# Feedback buttons for last assistant message
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
):
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key="thumbs_up", help="Good answer"):
            try:
                st.session_state.agent.improvement_engine.record_feedback(
                    st.session_state.messages[-2]["content"], "positive"
                )
                st.session_state.agent.memory.update_user_profile(
                    "default", {"thumbs_up": True}
                )
                st.toast("Thanks for the feedback! 👍", icon="✅")
            except Exception:
                pass
    with col2:
        if st.button("👎", key="thumbs_down", help="Bad answer"):
            try:
                st.session_state.agent.improvement_engine.record_feedback(
                    st.session_state.messages[-2]["content"], "negative"
                )
                st.toast(
                    "Sorry about that. You can correct me by saying "
                    "'Actually, the answer is...'",
                    icon="📝",
                )
            except Exception:
                pass


# ======================================================================
# Chat input
# ======================================================================

user_input = st.chat_input(
    "Ask me anything — I'll automatically research and verify the answer"
)

if user_input or (
    st.session_state.get("realtime_audio")
    and not st.session_state.get("realtime_submitted")
):
    # The browser Web Speech API already injected & submitted the text
    # directly into the chat input, so user_input is always populated.
    # The realtime_audio path below is kept only as a fallback for the
    # old audio-recorder-streamlit path (now removed).
    audio_bytes = None
    if audio_file is not None:
        audio_bytes = audio_file.read()

    result = process_input(user_input or "", audio_bytes=audio_bytes)
    processed_text = result["processed_text"] or user_input or ""

    for action in result["actions_taken"]:
        st.toast(action, icon="✅")

    # User message
    st.session_state.messages.append({"role": "user", "content": processed_text})
    with st.chat_message("user", avatar=CHAT_AVATAR_USER):
        st.markdown(processed_text)

    # Build history
    conv_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
    ]

    # Agent response
    with st.chat_message("assistant", avatar=CHAT_AVATAR_AI):
        confidence_placeholder = st.empty()
        with st.spinner("🧠 Thinking…"):
            agent_result = st.session_state.agent.ask(
                processed_text,
                conv_history,
                active_doc_sites=st.session_state.active_doc_sites,
            )

        # Show a subtle note if auto deep research was used
        if "auto_deep_research" in agent_result.get("enhanced_features_used", []):
            confidence_placeholder.info(
                "🔍 I searched the web to find a more complete answer for you.",
                icon="🌐",
            )
        else:
            confidence_placeholder.empty()

        answer = agent_result.get("final_answer", "Sorry, something went wrong.")
        st.markdown(answer, unsafe_allow_html=True)

        meta = {
            "confidence": agent_result.get("confidence"),
            "sources": agent_result.get("sources", []),
            "web_search": agent_result.get("web_search_performed", False),
            "follow_ups": agent_result.get("follow_up_suggestions", []),
            "fact_check_report": agent_result.get("fact_check_report"),
            "features": agent_result.get("enhanced_features_used", []),
            "response_time_ms": agent_result.get("response_time_ms"),
            "conversation_state": agent_result.get("conversation_state"),
            "quality_score": agent_result.get("quality_score"),
        }
        render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )
    st.rerun()
