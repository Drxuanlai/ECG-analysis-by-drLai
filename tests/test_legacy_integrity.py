"""Confirm the frozen baseline can be parsed and contains critical functions."""
import ast
from pathlib import Path

def test_legacy_source_parses_and_has_critical_entrypoints():
    p = Path(__file__).resolve().parents[1] / 'legacy' / 'app_v17_1_original.py'
    tree = ast.parse(p.read_text(encoding='utf-8'))
    names = {n.name for n in tree.body if isinstance(n,(ast.FunctionDef, ast.AsyncFunctionDef))}
    for n in ('analyze_gemini','run_with_failover','_v171_grid_suppress','v17_segmentation_overlay'):
        assert n in names, f'Missing critical function: {n}'
