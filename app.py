"""Streamlit Cloud entry point — loads the main app from app/app.py."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("qt23_app", ROOT / "app" / "app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
