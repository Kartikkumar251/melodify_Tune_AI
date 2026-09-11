"""
BeatFlow AI — Unified Application Server Runner
Launches the FastAPI backend on http://0.0.0.0:8000
"""
import sys
from pathlib import Path

# Add backend and root directories to sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

for p in [str(BACKEND_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn
from backend.api_server import app

if __name__ == "__main__":
    print("=" * 65)
    print("  BEATFLOW AI  |  DAW & Music Synthesis Platform")
    print("  Server URL   :  http://localhost:8000")
    print("  Studio DAW   :  http://localhost:8000/ui/studio.html")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000)
