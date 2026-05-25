"""
src/agent/retriever.py
- rewrite_query(): rewrites user questions
- retrieve(): fetches chunks from vector store and performs hybrid keyword re-ranking
"""

import json
import re
from src.config import llm_completion
from src.storage.vector_store import VectorStore

# Basic English stopwords list to filter out noise from keyword matching
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have",
    "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him",
    "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt",
    "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such",
    "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres",
    "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent",
    "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom",
    "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve",
    "your", "yours", "yourself", "yourselves"
}

def tokenize(text: str) -> set[str]:
    """Helper to lowercase, remove non-alphanumeric chars, and filter stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    return {w for w in words if w not in STOPWORDS}

def calculate_keyword_score(query_tokens: set[str], chunk_text: str) -> float:
    """Computes exact keyword token overlap score between query and chunk."""
    if not query_tokens:
        return 0.0
    chunk_tokens = tokenize(chunk_text)
    overlap = query_tokens.intersection(chunk_tokens)
    # Return percentage of query keywords present in this chunk
    return len(overlap) / len(query_tokens)

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
    
    # Step 1: Tokenize the original query strings for exact keyword ranking
    query_tokens = set()
    for q in queries:
        query_tokens.update(tokenize(q))
    
    all_chunks = []
    seen = set()
    
    # Step 2: Fetch a larger candidate pool (top_k * 2) from vector DB
    candidate_k = top_k * 2
    for q in queries:
        vec = embed_query(q)
        results = vector_store.search_chunks_by_prefix(vec, site_prefix=session_id, top_k=candidate_k)
        for r in results:
            if r["text"] not in seen:
                seen.add(r["text"])
                all_chunks.append(r)
                
    # Step 3: Run Jaccard token-overlap Hybrid Re-ranking
    for chunk in all_chunks:
        semantic_score = chunk.get("score", 0.0)
        # Compute exact token overlap score
        keyword_score = calculate_keyword_score(query_tokens, chunk["text"])
        
        # Combine: Semantic Score + 0.3 * Keyword Score
        # (This ensures exact keyword matches boost a chunk's ranking)
        chunk["hybrid_score"] = semantic_score + (0.3 * keyword_score)
        chunk["keyword_overlap"] = keyword_score
        
    # Step 4: Sort candidate pool by hybrid score and return final top_k
    all_chunks.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    
    return all_chunks[:top_k]
