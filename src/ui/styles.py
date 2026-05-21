import streamlit as st

def apply_custom_css():
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
            border: 1px solid rgba(99, 102, 241, 0.3);
            white-space: nowrap;
        }
        
        .feature-badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            margin-right: 4px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .fact-verified { color: #10b981; font-weight: 500; font-size: 0.85rem; }
        .fact-unverified { color: #f59e0b; font-weight: 500; font-size: 0.85rem; }
        .fact-contradicted { color: #ef4444; font-weight: 500; font-size: 0.85rem; }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #6366f1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
