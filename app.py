"""ECG Core v1.1: legacy app, followed by isolated adaptive QA panel."""
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'legacy'/'app_v17_1_original.py'),run_name='__main__')
from ui.adaptive_panel import render
render()

from ui.grid_aware_panel import render as render_grid_aware
render_grid_aware()

from ui.layout_panel import render as render_adaptive_layout
render_adaptive_layout()
