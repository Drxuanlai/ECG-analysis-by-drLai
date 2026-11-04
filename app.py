"""ECG Core v1.5: original clinical reader + one consolidated research QA entry.

Legacy clinical code and API/failover are untouched. Old QA is opt-in.
"""
from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / 'legacy' / 'app_v17_1_original.py'), run_name='__main__')
from ui.multi_layout_panel import render as render_ecg_core_qa
render_ecg_core_qa()
