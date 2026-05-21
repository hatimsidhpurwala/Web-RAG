import streamlit as st
from typing import List

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
