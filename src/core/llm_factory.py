"""
Factory for dynamically generating LangChain LLM instances based on Streamlit UI selection.
"""

import os
import streamlit as st
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import LLM_MODEL

logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.0, max_tokens: int = 2000) -> BaseChatModel:
    """
    Returns the selected LangChain chat model instance based on Streamlit session state.
    Defaults to Groq if no state is present (e.g. during background execution without UI).
    Falls back to Groq automatically if the selected provider fails.
    """
    provider = "Groq"
    if "llm_provider" in st.session_state:
        provider = st.session_state.llm_provider
        
    api_key = ""

    try:
        if provider == "Groq":
            from langchain_groq import ChatGroq
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY", "")
            return ChatGroq(
                api_key=api_key,
                model=LLM_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "Google Gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-2.5-flash",
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        elif provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY", "")
            return ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "Anthropic":
            from langchain_anthropic import ChatAnthropic
            if not api_key:
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
            return ChatAnthropic(
                api_key=api_key,
                model="claude-3-5-sonnet-latest",
                temperature=temperature,
                max_tokens=max_tokens,
            )

    except Exception as exc:
        logger.error(
            "Provider '%s' failed: %s — falling back to Groq automatically.",
            provider, exc,
        )

    # ── Universal fallback: always Groq ──────────────────────────
    from langchain_groq import ChatGroq
    fallback_key = os.getenv("GROQ_API_KEY", "")
    logger.warning("Using Groq fallback (provider '%s' unavailable).", provider)
    return ChatGroq(
        api_key=fallback_key,
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
