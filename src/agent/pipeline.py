"""
src/agent/pipeline.py
Main entry point — routes chat/RAG/web, calls other agent files
"""

import logging
from typing import Optional

from src.storage.vector_store import VectorStore
from src.storage.session import get_history, save_message
from src.agent.retriever import rewrite_query, retrieve
from src.agent.web_search import search, fetch_and_embed
from src.agent.responder import generate
from src.config import llm_completion

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.vector_store = VectorStore()

    def ask(self, question: str, session_id: str, uploaded_docs: bool = False, provider: str = "google") -> str:
        history = get_history(session_id)
        
        # Inline routing logic: Greeting check
        prompt = f"Is this a simple greeting or farewell? Question: '{question}'. Answer with YES or NO."
        try:
            res_text = llm_completion(
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            is_greeting = "YES" in res_text.upper()
        except Exception as e:
            logger.error(f"Routing check failed: {e}")
            is_greeting = False

        if is_greeting:
            try:
                answer = llm_completion(
                    provider=provider,
                    messages=[{"role": "user", "content": question}],
                    temperature=0.3
                )
                if not answer: answer = "Hello!"
                save_message(session_id, "user", question)
                save_message(session_id, "assistant", answer)
                return answer
            except Exception as e:
                return f"Error: {e}"

        queries = rewrite_query(question, history, provider=provider)
        chunks = []
        
        if uploaded_docs:
            chunks = retrieve(queries, session_id, self.vector_store, top_k=5)
            
        # If no docs or chunks are insufficient, fall back to web search
        if not chunks:
            urls = search(queries[0], max_results=3)
            for url in urls:
                fetch_and_embed(url, session_id, self.vector_store)
            chunks = retrieve(queries, session_id, self.vector_store, top_k=5)

        answer = generate(question, chunks, history, provider=provider)
        
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        
        return answer
