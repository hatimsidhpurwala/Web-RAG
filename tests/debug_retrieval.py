"""Debug script: diagnose why PDF source filter is not working."""
import sys; sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv("config/.env")
from src.database.vector_store import VectorStore
from src.core.embedder import embed_query

vs = VectorStore()
q = embed_query("what is the pdf about which i have uploaded right now")

print("=== ALL SOURCES (threshold=0.6) ===")
all_results = vs.search_chunks(q, top_k=8, score_threshold=0.6)
for r in all_results:
    print(f"  score={r['score']:.3f}  site={r['site_name'][:50]}")

print("\n=== PDF ONLY (threshold=0.6) ===")
pdf_thresh = vs.search_chunks_by_prefix(q, "pdf_", top_k=5, score_threshold=0.6)
print(f"  Results count: {len(pdf_thresh)}")
for r in pdf_thresh:
    print(f"  score={r['score']:.3f}  site={r['site_name'][:50]}")

print("\n=== PDF ONLY (no threshold) ===")
pdf_nothresh = vs.search_chunks_by_prefix(q, "pdf_", top_k=5, score_threshold=None)
print(f"  Results count: {len(pdf_nothresh)}")
for r in pdf_nothresh:
    print(f"  score={r['score']:.3f}  site={r['site_name'][:50]}")

print("\n=== SETTINGS ===")
from config.settings import SIMILARITY_THRESHOLD, TOP_K_PER_QUERY, TOP_K_RESULTS
print(f"  SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
print(f"  TOP_K_PER_QUERY = {TOP_K_PER_QUERY}")
print(f"  TOP_K_RESULTS = {TOP_K_RESULTS}")
