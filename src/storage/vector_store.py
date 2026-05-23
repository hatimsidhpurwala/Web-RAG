"""
src/storage/vector_store.py
Qdrant vector-store CRUD operations + Embeddings.
"""

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION_NAME, EMBEDDING_DIMENSION, QDRANT_API_KEY, QDRANT_LOCAL_DIR, QDRANT_URL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_query(query: str) -> List[float]:
    return get_embedder().encode(query, convert_to_numpy=True).tolist()

def embed_chunks(chunks: List[dict]) -> List[dict]:
    if not chunks: return chunks
    texts = [c["text"] for c in chunks]
    vectors = get_embedder().encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    for c, v in zip(chunks, vectors):
        c["embedding"] = v.tolist()
    return chunks

class VectorStore:
    _shared_client: Optional[QdrantClient] = None

    def __init__(self) -> None:
        if VectorStore._shared_client is not None:
            self.client = VectorStore._shared_client
        elif QDRANT_URL and QDRANT_API_KEY:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
            VectorStore._shared_client = self.client
        else:
            QDRANT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(QDRANT_LOCAL_DIR))
            VectorStore._shared_client = self.client

        try:
            cols = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in cols:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
                )
        except Exception:
            pass

    def store_chunks_for_site(self, chunks: List[dict], site_name: str) -> int:
        if not chunks: return 0
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "source_url": chunk.get("source_url", ""),
                    "site_name": site_name,
                },
            )
            for chunk in chunks
        ]
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+100])
        return len(points)

    def search_chunks(self, query_vector: List[float], top_k: int = 8, site_name_filter: Optional[str] = None) -> List[dict]:
        query_filter = Filter(must=[FieldCondition(key="site_name", match=MatchValue(value=site_name_filter))]) if site_name_filter else None
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k, query_filter=query_filter
            )
        except Exception:
            return []
        return [{"text": h.payload.get("text", ""), "source_url": h.payload.get("source_url", ""), "site_name": h.payload.get("site_name", ""), "score": h.score} for h in results]

    def search_chunks_by_prefix(self, query_vector: List[float], site_prefix: str, top_k: int = 8) -> List[dict]:
        try:
            scroll_results, _ = self.client.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=["site_name"], with_vectors=False)
            matching_sites = sorted(set(p.payload["site_name"] for p in scroll_results if p.payload.get("site_name", "").startswith(site_prefix)))
        except Exception:
            return []
        
        if not matching_sites: return []
        
        all_chunks = []
        seen = set()
        for site in matching_sites:
            for c in self.search_chunks(query_vector, top_k, site):
                key = c["text"][:200]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(c)
        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        return all_chunks[:top_k]

    def clear_site(self, site_name: str) -> None:
        self.client.delete(collection_name=COLLECTION_NAME, points_selector=Filter(must=[FieldCondition(key="site_name", match=MatchValue(value=site_name))]))

    def has_site(self, site_name: str) -> bool:
        try:
            results = self.client.scroll(collection_name=COLLECTION_NAME, scroll_filter=Filter(must=[FieldCondition(key="site_name", match=MatchValue(value=site_name))]), limit=1, with_payload=False, with_vectors=False)
            return len(results[0]) > 0
        except Exception:
            return False
