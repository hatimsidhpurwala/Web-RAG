"""
RAGAgent – public API for the LangGraph-based RAG pipeline.

This module's single job is to expose the .ask() method that the
Streamlit app (and any other caller) uses.  It orchestrates:

  1. Pre-processing  – language, sentiment, conversation context,
                       query strategy, memory checks
  2. Graph execution – delegates to builder.build_graph()
  3. Post-processing – auto deep research, fact-check, sentiment
                       adaptation, personalisation, translation,
                       metrics recording
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.agents.graph.builder import build_graph
from src.agents.graph.state import AgentState

from src.agents.query_analyzer import analyze_query
from src.agents.memory import AdaptiveMemory
from src.agents.fact_verifier import FactVerifier
from src.agents.conversation_manager import ConversationState
from src.agents.clarification_agent import ClarificationAgent
from src.agents.sentiment_adapter import SentimentAdapter
from src.agents.multilingual import MultilingualHandler
from src.agents.self_improvement import SelfImprovementEngine
from src.agents.agent_team import AgentTeam

from src.agents.query_generator import generate_queries
from src.agents.response_generator import generate_response
from src.agents.retriever import retrieve_chunks
from src.agents.web_searcher import deep_research

from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGAgent:
    """High-level agent that exposes a single `.ask()` method."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.memory = AdaptiveMemory()
        self.fact_verifier = FactVerifier()
        self.conversation_state = ConversationState()
        self.clarification_agent = ClarificationAgent()
        self.sentiment_adapter = SentimentAdapter()
        self.multilingual = MultilingualHandler()
        self.improvement_engine = SelfImprovementEngine()
        self.agent_team = AgentTeam(self.vector_store)
        self.graph = build_graph(self.vector_store)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        conversation_history: Optional[List[dict]] = None,
        user_id: str = "default",
        enable_fact_check: bool = True,
        active_doc_sites: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process *question* through the full enhanced RAG pipeline.

        Parameters
        ----------
        question : str
            Raw user input (any language).
        conversation_history : list[dict] | None
            Previous turns as ``{"role": …, "content": …}`` dicts.
        user_id : str
            Per-user personalisation key.
        enable_fact_check : bool
            Set False to skip fact-verification (faster).
        active_doc_sites : list[str] | None
            Site names of PDFs/docs uploaded in this session
            (e.g. ``["pdf_Datasheet-SALTO"]``).  When provided and the
            question appears to be about an uploaded document, retrieval
            is restricted to ``pdf_*`` sources so web content is ignored.
        """
        start_time = time.time()

        # ── 1. Memory: return known correction immediately ──────────────
        self.memory.track_query(question)
        correction = self.memory.get_correction(question)
        if correction:
            logger.info("Returning known correction for: %s", question[:60])
            return {
                "question": question,
                "final_answer": f"📝 *Updated answer based on previous feedback:*\n\n{correction}",
                "confidence": 0.95,
                "sources": ["user_correction"],
                "follow_up_suggestions": [],
                "web_search_performed": False,
                "enhanced_features_used": ["memory_correction"],
                "response_time_ms": 0,
            }

        # ── 2. Pre-processing ───────────────────────────────────────────
        lang_info = self.multilingual.process_query(question)
        working_question = lang_info["english_query"]

        sentiment = self.sentiment_adapter.analyze_sentiment(
            question, conversation_history
        )

        conv_state = self.conversation_state.analyze_message(
            working_question, conversation_history or []
        )

        # Learn from corrections ("Actually, the answer is …")
        if conv_state.get("is_correction") and conversation_history:
            last_q, last_a = "", ""
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant" and not last_a:
                    last_a = msg["content"]
                elif msg["role"] == "user" and not last_q:
                    last_q = msg["content"]
                if last_q and last_a:
                    break
            if last_q and last_a:
                self.memory.learn_from_correction(
                    query=last_q,
                    wrong_answer=last_a,
                    correct_answer=question,
                    user_id=user_id,
                )

        enhanced_question = conv_state.get("enhanced_query", working_question)
        strategy = analyze_query(enhanced_question, conversation_history)

        # ── 2b. Short follow-up enrichment ──────────────────────────────
        # When user sends a vague 1-3 word continuation like "explain me",
        # "continue", "tell me more" while active docs exist, auto-expand
        # the question using the last user turn to prevent the intent
        # classifier from treating it as a direct/greeting response.
        _SHORT_FOLLOWUPS = {
            "explain me", "explain", "continue", "go on", "tell me more",
            "more details", "elaborate", "summarize", "summary", "details",
            "ok", "okay", "yes", "yeah", "sure", "and", "what else",
            "more", "next", "show me", "give me more", "expand",
        }
        q_stripped = enhanced_question.strip().lower().rstrip(".")
        is_short_followup = q_stripped in _SHORT_FOLLOWUPS or (
            len(enhanced_question.split()) <= 3 and
            not any(c.isdigit() for c in enhanced_question)
        )

        if is_short_followup and active_doc_sites and conversation_history:
            # Find the last user question from history
            last_user_q = ""
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    last_user_q = msg.get("content", "")
                    break
            if last_user_q:
                enhanced_question = f"{last_user_q} — please provide a detailed explanation"
                logger.info(
                    "Short follow-up '%s' expanded to: '%s'",
                    working_question, enhanced_question
                )

        # ── 3. Detect document questions → restrict retrieval to pdf_ ───
        _doc_keywords = (
            "pdf", "document", "uploaded", "file", "datasheet",
            "the data", "this data", "above data", "attached",
            "right now", "this file", "the file", "i sent", "i shared",
            "just uploaded", "doc", "specification", "spec sheet",
        )
        _vague_doc_phrases = (
            "what is it about", "summarize", "tell me about",
            "what does it say", "what is this about", "what is the pdf",
            "explain the", "explain this", "overview of",
        )
        has_active_docs = bool(active_doc_sites)
        q_lower = enhanced_question.lower()
        is_document_q = any(kw in q_lower for kw in _doc_keywords)
        if has_active_docs and any(p in q_lower for p in _vague_doc_phrases):
            is_document_q = True

        # When multiple PDFs are active, check if the user named a specific one
        # e.g. "the SALTO keycard pdf" → restrict to pdf_Datasheet-SALTO...
        specific_prefix: Optional[str] = None
        if has_active_docs and len(active_doc_sites) > 1:
            for site in active_doc_sites:
                # Strip "pdf_" prefix to get the filename stem
                stem = site.replace("pdf_", "").lower()
                # Check if any significant word (>4 chars) from the stem appears in the question
                stem_words = [w for w in stem.replace("-", " ").replace("_", " ").split() if len(w) > 4]
                if any(word in q_lower for word in stem_words):
                    specific_prefix = site  # exact site match → filter to that one doc
                    logger.info("Specific PDF referenced: %s", site)
                    break

        # source_prefix="pdf_" tells the retriever node to filter by that prefix
        # specific_prefix is an exact site_name (used for single-doc questions with multiple active)
        source_prefix: Optional[str] = (
            "pdf_" if (is_document_q and has_active_docs) else None
        )
        if source_prefix:
            logger.info(
                "Document question detected – restricting retrieval to '%s' "
                "(active docs: %s)", source_prefix, active_doc_sites,
            )


        # ── 4. Build initial state and run graph ────────────────────────
        initial_state: AgentState = {
            "question": enhanced_question,
            "original_question": question,
            "conversation_history": conversation_history or [],
            "web_search_performed": False,
            "query_strategy": strategy.model_dump(),
            "sentiment": sentiment,
            "conversation_state": conv_state,
            "language_info": lang_info,
            "enhanced_features_used": [],
        }
        if source_prefix:
            initial_state["source_prefix"] = source_prefix

        try:
            result = self.graph.invoke(initial_state)
        except Exception as exc:
            logger.error("Graph invocation failed: %s", exc, exc_info=True)
            result = {
                "question": question,
                "final_answer": "I'm sorry, I encountered an error. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "follow_up_suggestions": [],
                "web_search_performed": False,
                "enhanced_features_used": [],
            }

        # ── 5. Post-processing ──────────────────────────────────────────
        final_answer = result.get("final_answer", "")
        features_used = list(result.get("enhanced_features_used", []))
        confidence = result.get("confidence", 0.0)

        # 5a. Auto deep research – triggered only for retrieval-type queries
        # NEVER triggered for: greetings, gratitude, farewell, clarification,
        # or very short/vague queries (< 4 words).
        _NON_RESEARCH_INTENTS = {"greeting", "gratitude", "farewell", "clarification"}
        _is_pure_doc_q = is_document_q and not any(
            kw in q_lower for kw in (
                "distributor", "dealer", "reseller", "supplier", "partner",
                "contact", "phone", "email", "address", "office", "where can i",
                "where to buy", "where to find", "price", "cost",
            )
        )
        _intent = result.get("intent", "retrieval_needed")
        _q_word_count = len(question.strip().split())
        if (
            confidence < 0.90
            and not result.get("web_search_performed", False)
            and not _is_pure_doc_q
            and _intent not in _NON_RESEARCH_INTENTS
            and _q_word_count >= 4            # skip vague 1-3 word queries
        ):
            logger.info(
                "Confidence %.2f < 0.90 – auto deep research (intent=%s): %s",
                confidence, _intent, enhanced_question[:80],
            )
            try:
                research = self._deep_research(enhanced_question, conversation_history)
                # Always adopt web research answer if web search was run —
                # even if confidence didn't improve, raw web data is better
                # than a vague 'info not available' answer.
                if research.get("web_search_performed"):
                    final_answer = research["final_answer"]
                    confidence = max(research.get("confidence", 0), confidence)
                    result["sources"] = research.get("sources", [])
                    result["web_search_performed"] = True
                    result["confidence"] = confidence
                    # Preserve raw DDG cards so Streamlit can render them
                    result["research_info"] = research.get("research_info", {})
                    features_used.append("auto_deep_research")
            except Exception as exc:
                logger.error("Auto deep research failed: %s", exc)

        # 5b. Fact verification
        if enable_fact_check and result.get("retrieved_chunks"):
            try:
                fc = self.fact_verifier.verify_answer(
                    final_answer,
                    result.get("retrieved_chunks", []),
                    enhanced_question,
                )
                result["fact_check_report"] = {
                    "overall_reliability": fc.overall_reliability,
                    "verification_summary": fc.verification_summary,
                    "contradictions": fc.contradictions,
                    "claims_count": len(fc.claims),
                }
                if fc.verified_answer:
                    final_answer = fc.verified_answer
                features_used.append("fact_verification")
            except Exception as exc:
                logger.error("Fact verification failed: %s", exc)

        # 5c. Sentiment adaptation
        final_answer = self.sentiment_adapter.adapt_response(final_answer, sentiment)
        features_used.append("sentiment_adaptation")

        # 5d. Per-user personalisation
        final_answer = self.memory.personalize_response(user_id, final_answer)

        # 5e. Translate back if needed
        if lang_info.get("needs_translation"):
            final_answer = self.multilingual.translate_response(
                final_answer, lang_info["source_language"]
            )
            features_used.append("translation")

        # ── 6. Finalise ─────────────────────────────────────────────────
        result["final_answer"] = final_answer
        result["confidence"] = confidence
        result["enhanced_features_used"] = features_used

        elapsed_ms = int((time.time() - start_time) * 1000)
        result["response_time_ms"] = elapsed_ms

        try:
            self.improvement_engine.record_interaction(
                query=question,
                answer=final_answer[:500],
                confidence=confidence,
                response_time_ms=elapsed_ms,
                web_search_used=result.get("web_search_performed", False),
                sources=result.get("sources", []),
            )
        except Exception:
            pass

        self.memory.update_user_profile(user_id, {
            "topic": conv_state.get("current_topic", ""),
        })

        return result

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _deep_research(
        self,
        topic: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Run deep web research and return a fresh answer."""
        logger.info("Deep research (auto): %s", topic[:80])
        try:
            info = deep_research(topic, self.vector_store)
        except Exception as exc:
            logger.error("Deep research failed: %s", exc)
            info = {"queries_used": [], "sites_indexed": [], "total_chunks": 0, "raw_results": []}

        # Build synthetic chunks from raw DDG result cards so the LLM
        # always sees the actual web-found URLs / company names / snippets.
        raw_results: list = info.get("raw_results", [])
        synthetic_chunks: list[dict] = []
        for r in raw_results[:6]:
            title   = r.get("title", "")
            url     = r.get("url", "")
            snippet = r.get("snippet", "")
            if not (title or snippet):
                continue
            synthetic_chunks.append({
                "text": (
                    f"**{title}**\n"
                    f"Website: {url}\n\n"
                    f"{snippet[:500]}"
                ),
                "source_url":     url,
                "site_name":      "web_search_result",
                "score":          0.75,
                "context_header": "Web Search Result",
                # Pass through scraped contact details
                "phones":    r.get("phones", []),
                "emails":    r.get("emails", []),
                "addresses": r.get("addresses", []),
                "whatsapp":  r.get("whatsapp", []),
            })

        queries_obj = generate_queries(topic, conversation_history)
        kb_chunks   = retrieve_chunks(queries_obj.queries, self.vector_store)

        # Prepend synthetic web cards so LLM sees live results first
        all_chunks = synthetic_chunks + kb_chunks[:6]
        resp = generate_response(topic, all_chunks, conversation_history)

        return {
            "question": topic,
            "final_answer": resp.answer,
            "confidence": resp.confidence,
            "sources": [s.source_url for s in resp.sources_used],
            "follow_up_suggestions": resp.follow_up_suggestions or [],
            "web_search_performed": True,
            "research_info": info,
            "enhanced_features_used": ["auto_deep_research"],
        }

    # ──────────────────────────────────────────────────────────────────
    # Sidebar helpers
    # ──────────────────────────────────────────────────────────────────

    def get_performance_stats(self) -> dict:
        try:
            return self.improvement_engine.analyze_performance()
        except Exception:
            return {}

    def get_memory_stats(self) -> dict:
        try:
            return {
                **self.memory.get_stats(),
                "top_queries": self.memory.get_top_queries(5),
            }
        except Exception:
            return {}
