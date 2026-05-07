"""
Response generator agent.

Synthesises a final answer from retrieved context chunks.
Uses with_structured_output for type-safe, hallucination-resistant responses.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import ResponseGeneration

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise, factual RAG assistant. Answer the user's question using
the provided context chunks. Follow these rules strictly.

═══════════════════════════════════════════════════════
RULE 1 — ALWAYS SHOW WHAT YOU FOUND (most critical rule)
═══════════════════════════════════════════════════════
  NEVER say:
    - "The available information does not cover..."
    - "Based on general knowledge, you can visit their website"
    - "The context does not include..."
    - "I cannot find a specific list..."
  ALWAYS present whatever IS in the context chunks — even if partial.
  Only if chunks are COMPLETELY EMPTY, say: "No data was found for this query."

═══════════════════════════════════════════════════════
RULE 2 — WEB SEARCH RESULTS (treat as primary source)
═══════════════════════════════════════════════════════
  Chunks labelled "Web Search Result" are LIVE web data.
  For EVERY such chunk, output:

    **[Company / Page Title]**
    🌐 [URL]
    📝 [full snippet text from the chunk]

  List ALL of them. Do NOT summarize or skip any URL or company name.
  Do NOT say "visit their website" — output the actual URL every time.

═══════════════════════════════════════════════════════
RULE 3 — DISTRIBUTOR / CONTACT / LOCATION QUESTIONS
═══════════════════════════════════════════════════════
  Extract EVERY company name, phone, email, address, URL from context.
  Format as a structured list — NOT a paragraph.
  If geography is partially covered (e.g. UAE found but India asked):
    - Still list what was found
    - Add: "📌 Find [India] partners at: https://maps.salto.systems/"

═══════════════════════════════════════════════════════
RULE 4 — DOCUMENT QUESTIONS ("summarize", "explain the pdf")
═══════════════════════════════════════════════════════
  Extract ALL details: specs, models, features, contacts, certifications.
  Use bold section headers matching the document structure.
  Confidence: 0.85 when context covers the document well.

═══════════════════════════════════════════════════════
RULE 5 — FORMAT
═══════════════════════════════════════════════════════
  Use **bold** for headings and key terms.
  Use bullet points (–) for lists of 3+ items.
  Keep paragraphs under 4 sentences.
  Never cite chunk numbers ("According to chunk 2...").
  Never say "As an AI" or refer to yourself.

═══════════════════════════════════════════════════════
RULE 6 — CONFIDENCE SCORING
═══════════════════════════════════════════════════════
  0.90–1.0  Direct, complete answer from context
  0.75–0.89 Partial answer; small gaps noted
  0.50–0.74 Thin context; answer supplemented
  0.30–0.49 Mostly general knowledge
  0.00–0.29 No relevant context at all

═══════════════════════════════════════════════════════
RULE 7 — FOLLOW-UP SUGGESTIONS
═══════════════════════════════════════════════════════
  Suggest exactly 2 specific follow-up questions related to the answer.
  BAD:  "Would you like to know more?"
  GOOD: "What are the memory options for the CCVD40xx model?"

═══════════════════════════════════════════════════════
RULE 8 — TABLE FORMAT FOR DISTRIBUTOR / CONTACT QUERIES
═══════════════════════════════════════════════════════
  Whenever the user asks about distributors, dealers, resellers,
  suppliers, contacts, or offices in any location, output a
  Markdown table with these EXACT columns:

  | Company / Website | Phone | Email | Address | Notes |
  |---|---|---|---|---|

  Rules for the table:
  - Extract data from ALL chunks including Web Search Result chunks.
  - Leave a cell blank (—) ONLY if the data is truly not present
    anywhere in the context for that specific row.
  - Do NOT put all companies in one row — one row per company/dealer.
  - After the table, add a "📌 Partner Finder" line with
    the official locator URL if one is available.
  - NEVER replace a table with a paragraph if data was found.

═══════════════════════════════════════════════════════
ABSOLUTE DO-NOTS
═══════════════════════════════════════════════════════
  ✗ NEVER write "The available information does not cover..."
  ✗ NEVER write "visit their website" without the actual URL.
  ✗ NEVER say "I cannot help" — always synthesize what was found.
  ✗ Do not hallucinate phone numbers, addresses, or prices.
  ✗ Do not repeat the same info twice.
  ✗ Do not start with "Certainly!", "Of course!", "Great question!".
"""

# ── Structured LLM ──────────────────────────────────────────────────────────
# max_tokens capped at 1200 to stay well within Groq's 64K context window
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.2,
    max_tokens=1200,
)
_structured_llm = _llm.with_structured_output(ResponseGeneration)

# Safety limits — prevents context overflow
_MAX_CHUNKS      = 12    # max chunks forwarded to LLM
_MAX_CHUNK_CHARS = 700   # chars per chunk text (trimmed if longer)
_MAX_HISTORY     = 4     # last N conversation messages included

# Keywords that flag a contact/distributor query → trigger table formatting
_TABLE_KEYWORDS = (
    "distributor", "dealer", "reseller", "supplier", "partner",
    "contact details", "phone number", "email address",
    "where to buy", "where can i find", "where can i get",
    "who sell", "who deal", "agent", "authorised", "authorized",
    "stockist", "vendor", "office address", "contact info",
)

# At least one of these must ALSO appear for the query to be "specific enough"
# to trigger table/contact formatting — prevents "can i help" matching
_SPECIFICITY_KEYWORDS = (
    "salto", "card", "keycard", "rfid", "access", "lock",
    "uae", "dubai", "india", "africa", "qatar", "saudi", "uk", "usa",
    "price", "buy", "purchase", "distributor", "dealer", "supplier",
    "reseller", "vendor", "stockist", "partner", "office",
)


# ── Table builder ────────────────────────────────────────────────────────────

def _build_table_from_chunks(chunks: List[dict], llm_answer: str) -> str:
    """Build a guaranteed Markdown table from web-result chunks + LLM answer.

    Uses a fast LLM call with explicit JSON schema to extract structured rows,
    then renders them as a Markdown table. Falls back to the original answer
    if extraction fails or produces no rows.
    """
    from groq import Groq as _Groq
    import json as _json

    # Gather all web result chunks
    web_chunks = [c for c in chunks if c.get("context_header") == "Web Search Result"]
    if not web_chunks:
        return llm_answer  # nothing to tabulate

    # Build a compact context for extraction
    compact_parts: list[str] = []
    for c in web_chunks[:8]:
        title   = c.get("text", "").split("\n")[0].replace("**", "").strip()
        url     = c.get("source_url", "")
        phones  = "; ".join(c.get("phones", []))
        emails  = "; ".join(c.get("emails", []))
        addrs   = "; ".join(c.get("addresses", [])[:1])
        wapp    = "; ".join(c.get("whatsapp", []))
        snippet = c.get("text", "")[:300]
        compact_parts.append(
            f"SOURCE: {url}\n"
            f"TITLE: {title}\n"
            f"PHONES: {phones or 'unknown'}\n"
            f"EMAILS: {emails or 'unknown'}\n"
            f"ADDRESS: {addrs or 'unknown'}\n"
            f"WHATSAPP: {wapp or 'unknown'}\n"
            f"SNIPPET: {snippet}"
        )
    compact_block = "\n---\n".join(compact_parts)

    extraction_prompt = (
        "Extract distributor/company information and return JSON.\n\n"
        "For each company/website found in the sources below, produce one object.\n"
        "Return ONLY valid JSON:\n"
        '{"rows": [{"company": "", "url": "", "phone": "", "email": "", '
        '"address": "", "whatsapp": "", "notes": ""}, ...]}\n\n'
        "Rules:\n"
        "- Use empty string for unknown fields (do NOT make up data).\n"
        "- Extract phones/emails exactly as written.\n"
        "- 'notes' = brief description from snippet (max 100 chars).\n\n"
        f"SOURCES:\n{compact_block}\n\n"
        f"ALSO USE THIS ANSWER FOR ADDITIONAL CONTEXT:\n{llm_answer[:600]}"
    )

    try:
        client = _Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        data = _json.loads(resp.choices[0].message.content)
        rows = data.get("rows", [])
    except Exception as exc:
        logger.warning("Table extraction failed: %s", exc)
        return llm_answer

    if not rows:
        return llm_answer

    # Render as Markdown table
    cols = ["Company / Website", "Phone", "Email", "Address", "WhatsApp", "Notes"]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"

    table_rows: list[str] = []
    for r in rows:
        url     = r.get("url", "").strip()
        company = r.get("company", "").strip() or url
        name_cell = f"[{company}]({url})" if url else (company or "—")
        cells = [
            name_cell,
            r.get("phone", "").strip()    or "—",
            r.get("email", "").strip()    or "—",
            r.get("address", "").strip()  or "—",
            r.get("whatsapp", "").strip() or "—",
            (r.get("notes", "").strip() or "—")[:100],
        ]
        # Escape pipe chars inside cells
        cells = [c.replace("|", "\\|") for c in cells]
        table_rows.append("| " + " | ".join(cells) + " |")

    table_md = f"{header}\n{sep}\n" + "\n".join(table_rows)
    logger.info("Built contact table with %d rows", len(table_rows))
    return table_md


# ── Public function ──────────────────────────────────────────────────────────

def generate_response(
    question: str,
    context_chunks: List[dict],
    conversation_history: Optional[List[dict]] = None,
) -> ResponseGeneration:
    """Generate a grounded response for *question* from *context_chunks*.

    Returns a guaranteed valid ResponseGeneration Pydantic object.
    For contact/distributor queries with web results, the answer is
    post-processed into a Markdown table regardless of LLM formatting choice.
    """
    # ── Cap & trim chunks; enrich Web Search Result chunks with contact data ──
    capped_chunks = context_chunks[:_MAX_CHUNKS]
    context_parts: list[str] = []
    for i, chunk in enumerate(capped_chunks, 1):
        source = chunk.get("source_url", "unknown")
        score  = chunk.get("score", 0)
        header = chunk.get("context_header", "")
        header_str = f" | section: {header}" if header else ""
        text   = chunk.get("text", "")

        # Append structured contact fields if this is a web result chunk
        if header == "Web Search Result":
            contact_lines: list[str] = []
            for ph in chunk.get("phones", []):
                contact_lines.append(f"Phone: {ph}")
            for em in chunk.get("emails", []):
                contact_lines.append(f"Email: {em}")
            for addr in chunk.get("addresses", [])[:2]:
                contact_lines.append(f"Address: {addr}")
            for wa in chunk.get("whatsapp", []):
                contact_lines.append(f"WhatsApp: {wa}")
            if contact_lines:
                text = text.rstrip() + "\n" + "\n".join(contact_lines)

        if len(text) > _MAX_CHUNK_CHARS:
            text = text[:_MAX_CHUNK_CHARS] + "…"
        context_parts.append(
            f"--- Chunk {i} (source: {source}, score: {score:.2f}{header_str}) ---\n"
            f"{text}\n"
        )
    context_block = (
        "\n".join(context_parts) if context_parts
        else "(no context retrieved — answer from general knowledge only, label it clearly)"
    )

    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Include last N conversation turns — capped to avoid token blowout
    if conversation_history:
        for msg in conversation_history[-_MAX_HISTORY:]:
            if msg.get("role") == "user":
                snippet = msg.get("content", "")[:300]
                messages.append(HumanMessage(content=f"[history]: {snippet}"))

    # Detect query type for post-processing
    q_lower = question.lower()
    is_contact_query = (
        any(kw in q_lower for kw in _TABLE_KEYWORDS)
        and any(kw in q_lower for kw in _SPECIFICITY_KEYWORDS)
    )
    has_web_results  = any(
        chunk.get("context_header") == "Web Search Result"
        for chunk in capped_chunks
    )

    user_content = (
        f"CONTEXT CHUNKS:\n{context_block}\n\n"
        f"USER QUESTION: {question}"
    )
    messages.append(HumanMessage(content=user_content))

    try:
        result: ResponseGeneration = _structured_llm.invoke(messages)
        logger.info(
            "Response generated (confidence=%.2f, sources=%d)",
            result.confidence,
            len(result.sources_used),
        )

        # ── Post-process: guarantee table format for contact queries ──────
        # This runs AFTER the LLM call so it's independent of whether
        # the LLM chose to follow formatting instructions.
        if is_contact_query and has_web_results:
            table_answer = _build_table_from_chunks(capped_chunks, result.answer)
            if table_answer != result.answer:
                result = ResponseGeneration(
                    answer="Here are the results found for your query:\n\n" + table_answer,
                    confidence=result.confidence,
                    sources_used=result.sources_used,
                    follow_up_suggestions=result.follow_up_suggestions,
                )

        return result

    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        return ResponseGeneration(
            answer=f"I encountered an error generating the response: {exc}",
            confidence=0.0,
            sources_used=[],
            follow_up_suggestions=None,
        )

