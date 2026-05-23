"""
run.py
Starts FastAPI and Streamlit
"""

import sys
import time
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    
    print("Starting Universal Web Scraper...")
    
    api_cmd = [sys.executable, "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
    ui_cmd = [sys.executable, "-m", "streamlit", "run", "src/api/ui.py", "--server.port", "8501"]
    
    try:
        api_proc = subprocess.Popen(api_cmd, cwd=root)
        ui_proc = subprocess.Popen(ui_cmd, cwd=root)
        
        print("\n=== SERVICES RUNNING ===")
        print("API: http://localhost:8000")
        print("Streamlit: http://localhost:8501")
        print("Press Ctrl+C to stop.\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        try: api_proc.terminate()
        except: pass
        try: ui_proc.terminate()
        except: pass
        print("Clean shutdown complete.")

if __name__ == "__main__":
    main()
