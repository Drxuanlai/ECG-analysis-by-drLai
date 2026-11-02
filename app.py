"""Streamlit entrypoint: compatibility mode for ECG Core v1.0.

Original v17.1 functionality stays unchanged while modules are extracted with tests.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "legacy" / "app_v17_1_original.py"), run_name="__main__")
