"""
run_all.py — Start both the FastAPI REST API and Streamlit UI together.

Usage:
    python run_all.py

This launches:
  - FastAPI  → http://localhost:8000   (REST API + interactive docs at /docs)
  - Streamlit → http://localhost:8501  (browser chat UI)

Both servers share the same Qdrant instance and LLM factory.
"""

import subprocess
import sys
import time
import threading
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_fastapi():
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "src.api.rest_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info",
        ],
        cwd=ROOT,
    )

def run_streamlit():
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "src/api/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
        ],
        cwd=ROOT,
    )

if __name__ == "__main__":
    print("=" * 60)
    print("  Smart RAG — Starting all services")
    print("  FastAPI  → http://localhost:8000/docs")
    print("  Streamlit→ http://localhost:8501")
    print("=" * 60)

    t_api = threading.Thread(target=run_fastapi,    daemon=True)
    t_ui  = threading.Thread(target=run_streamlit,  daemon=True)

    t_api.start()
    time.sleep(2)   # give FastAPI a head-start before Streamlit loads the model
    t_ui.start()

    try:
        t_api.join()
        t_ui.join()
    except KeyboardInterrupt:
        print("\nShutting down both servers...")
        sys.exit(0)
