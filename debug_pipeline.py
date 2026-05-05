"""
Full pipeline diagnostic: traces every step of what happens when
the user asks "list of distributors of SALTO keycards in Dubai".
"""
import sys; sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv("config/.env")

print("=" * 70)
print("STEP 1: DuckDuckGo search")
print("=" * 70)
from src.agents.web_searcher import search
results = search("SALTO keycards distributor Dubai UAE", max_results=5)
print(f"  Found {len(results)} results:")
for i, r in enumerate(results, 1):
    print(f"  [{i}] {r['title'][:60]}")
    print(f"      URL: {r['url']}")
    print(f"      Snippet: {r['snippet'][:100]}...")
    print()

print("=" * 70)
print("STEP 2: Query generation")
print("=" * 70)
from src.agents.query_generator import generate_queries
try:
    qg = generate_queries("list of distributors of SALTO keycards in Dubai")
    print(f"  query_type: {qg.query_type}")
    print(f"  queries: {qg.queries}")
    print(f"  entities: {qg.primary_entities}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 70)
print("STEP 3: search_and_scrape (the full pipeline)")
print("=" * 70)
from src.database.vector_store import VectorStore
from src.agents.web_searcher import search_and_scrape
vs = VectorStore()
info = search_and_scrape("SALTO CCVD20xx distributor Dubai", vs, max_results=5)
print(f"  sites_indexed: {info['sites_indexed']}")
print(f"  total_chunks: {info['total_chunks']}")
print(f"  raw_results count: {len(info.get('raw_results', []))}")
print()
for i, r in enumerate(info.get("raw_results", []), 1):
    print(f"  RAW [{i}] {r['title'][:60]}")
    print(f"         {r['url']}")
    print(f"         {r['snippet'][:100]}...")
    print()

print("=" * 70)
print("STEP 4: What the response generator would see")
print("=" * 70)
# Build synthetic chunks like node_re_generate_response does
synthetic = []
for r in info.get("raw_results", []):
    synthetic.append({
        "text": f"**{r['title']}**\nWebsite: {r['url']}\n\n{r['snippet']}",
        "source_url": r["url"],
        "score": 0.7,
    })
print(f"  Synthetic chunks that SHOULD be prepended: {len(synthetic)}")
for i, s in enumerate(synthetic, 1):
    print(f"  CHUNK [{i}]: {s['text'][:120]}...")
    print()
