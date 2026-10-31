"""
AudioBench Studio — Streamlit Application Entrypoint.

Provides immediate compatibility for Streamlit Community Cloud deployment
and local `streamlit run app.py` execution.
"""
import sys
from pathlib import Path

# Ensure src/ is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audiobench.app import main

if __name__ == "__main__":
    main()
