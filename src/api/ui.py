"""
src/api/ui.py
Entire Streamlit chat interface
"""

import sys
from pathlib import Path
import uuid
import streamlit as st
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.session import get_history

# ── Page config ──
st.set_page_config(page_title="Universal Web Scraper", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── Inline CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%); }
section[data-testid="stSidebar"] { background: rgba(15,12,41,0.95); border-right: 1px solid rgba(99,102,241,0.15); }
.stChatMessage { background: rgba(30,27,75,0.45) !important; border: 1px solid rgba(99,102,241,0.12); border-radius: 16px !important; backdrop-filter: blur(12px); padding: 1rem 1.25rem !important; margin-bottom: 0.75rem; }
.stChatInputContainer { background: rgba(15,12,41,0.8) !important; border-top: 1px solid rgba(99,102,241,0.15); }
.stChatInputContainer textarea { background: rgba(30,27,75,0.6) !important; border: 1px solid rgba(99,102,241,0.25) !important; border-radius: 12px !important; color: #e0e7ff !important; }
</style>""", unsafe_allow_html=True)

# ── Session Init ──
if "uploaded_docs" not in st.session_state: st.session_state.uploaded_docs = False

# ── Sidebar ──
with st.sidebar:
    st.markdown("<h2 style='color:#a5b4fc; text-align:center;'>Settings</h2>", unsafe_allow_html=True)
    
    phone_number = st.text_input("Enter Phone Number (Session ID)", value="default_user")
    st.session_state.session_id = phone_number.strip() if phone_number.strip() else "default_user"
    
    provider = st.selectbox("Select AI Model", ["google", "groq", "openai"], index=0)
    
    st.markdown("---")
    uploaded_files = st.file_uploader("Upload Files (PDF, Word, Images, Audio)", accept_multiple_files=True)
    if st.button("Process Uploads") and uploaded_files:
        with st.spinner("Processing..."):
            for f in uploaded_files:
                raw = f.read()
                files = {"file": (f.name, raw)}
                data = {"session_id": st.session_state.session_id}
                try:
                    res = requests.post("http://localhost:8000/api/upload", files=files, data=data)
                    if res.status_code == 200: st.success(f"Indexed {f.name}")
                    else: st.error(f"Failed {f.name}: {res.text}")
                except Exception as e:
                    st.error(f"API Error: {e}")
            st.session_state.uploaded_docs = True

    if st.button("Clear Session"):
        requests.delete(f"http://localhost:8000/api/session/{st.session_state.session_id}")
        st.session_state.uploaded_docs = False
        st.rerun()

# ── Main Chat ──
st.markdown("<h1 style='text-align:center;color:#a5b4fc;margin-bottom:0;'>🧠 Smart Web Scraper</h1>", unsafe_allow_html=True)

history = get_history(st.session_state.session_id)
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask me anything...")

# Audio Input below chat
audio_value = st.audio_input("Or record a voice message:")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": user_input,
                    "session_id": st.session_state.session_id,
                    "uploaded_docs": st.session_state.uploaded_docs,
                    "provider": provider
                }
                res = requests.post("http://localhost:8000/api/ask", json=payload)
                if res.status_code == 200:
                    answer = res.json().get("answer", "")
                else:
                    answer = f"Error: {res.text}"
            except Exception as e:
                answer = f"API Error: {e}"
        st.markdown(answer, unsafe_allow_html=True)
    st.rerun()

elif audio_value:
    with st.spinner("Processing voice message..."):
        try:
            files = {"audio": ("audio.wav", audio_value.read())}
            data = {
                "session_id": st.session_state.session_id,
                "provider": provider
            }
            res = requests.post("http://localhost:8000/api/voice", files=files, data=data)
            if res.status_code == 200:
                # The backend handles appending history, so we just rerun to fetch the latest history.
                st.rerun()
            else:
                st.error(f"Voice Error: {res.text}")
        except Exception as e:
            st.error(f"API Error: {e}")
