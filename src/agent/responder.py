"""
src/agent/responder.py
Generates final answer with citations
"""

from src.config import llm_completion

def generate(question: str, chunks: list[dict], history: list[dict], provider: str = "google") -> str:
    context_text = "\n\n".join(
        f"--- Source: {c.get('source_url', 'Unknown')} ---\n{c.get('text', '')}"
        for c in chunks
    )
    
    system_prompt = (
        "You are an AI assistant. Answer the user's question using the provided context chunks.\n"
        "Quote source chunks. Match user tone.\n"
        "If you don't know the answer based on the context, say so.\n"
        "Include the source URLs in your answer when referencing facts."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"
    })
    
    try:
        res_text = llm_completion(
            provider=provider,
            messages=messages,
            temperature=0.2
        )
        return res_text or "No response generated."
    except Exception as e:
        return f"Error generating response: {e}"
