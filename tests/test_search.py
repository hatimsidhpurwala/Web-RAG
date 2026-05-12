import sys; sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv("config/.env")
from src.agents.web_searcher import search

results = search("SALTO keycards distributor Dubai UAE", max_results=5)
print(f"Total results: {len(results)}")
for i, r in enumerate(results, 1):
    title = r["title"][:70]
    url = r["url"]
    snippet = r["snippet"][:120]
    print(f"[{i}] {title}")
    print(f"    URL: {url}")
    print(f"    Snippet: {snippet}")
    print()
