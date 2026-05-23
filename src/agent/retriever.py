"""
src/agent/retriever.py
- rewrite_query(): rewrites user questions
- retrieve(): fetches chunks from vector store
"""

import json
from src.config import llm_completion
from src.storage.vector_store import VectorStore

def rewrite_query(question: str, history: list, provider: str = "google") -> list[str]:
    prompt = f"""
    Given the user's question, generate 1 to 3 optimized search queries to retrieve relevant chunks from a vector database.
    Consider the conversation history context.
    Return ONLY a JSON array of strings, e.g. ["query 1", "query 2"].
    
    HISTORY: {history[-4:]}
    QUESTION: {question}
    """
    try:
        res_text = llm_completion(
            provider=provider,
            messages=[{"role": "user", "content": prompt}],
            json_mode=True
        )
        data = json.loads(res_text)
        queries = data.get("queries", [])
        if not queries and isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                    queries = v
                    break
        return queries if queries else [question]
    except Exception:
        return [question]

def retrieve(queries: list[str], session_id: str, vector_store: VectorStore, top_k: int = 5) -> list[dict]:
    from src.storage.vector_store import embed_query
    
    all_chunks = []
    seen = set()
    for q in queries:
        vec = embed_query(q)
        results = vector_store.search_chunks_by_prefix(vec, site_prefix=session_id, top_k=top_k)
        for r in results:
            if r["text"] not in seen:
                seen.add(r["text"])
                all_chunks.append(r)
                
    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_chunks[:top_k * 2]
