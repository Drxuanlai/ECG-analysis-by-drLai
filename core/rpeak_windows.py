"""ECG Core v1.5.3: windowed human R-peak annotation geometry.

Independent of Streamlit. x/y reference coordinates are always ROI-original
pixels; a window and/or display scale is a view transform only.
"""
from __future__ import annotations

import math
from typing import Any

VERSION = 'ecg-core-v1.5.3-windowed-annotation'


def make_windows(width: int, target_width: int = 640) -> list[dict[str, int]]:
    """Cover [0,width) exactly once, with reasonably sized windows."""
    width = int(width)
    target_width = int(target_width)
    if width < 1 or target_width < 80:
        raise ValueError('Invalid ROI width or target window size')
    n = max(1, math.ceil(width / target_width))
    # Equal width slices prevent a tiny final fragment.
    edges = [round(i * width / n) for i in range(n + 1)]
    return [dict(index=i, x_start=edges[i], x_end=edges[i+1], width=edges[i+1]-edges[i])
            for i in range(n)]


def window_for_x(windows: list[dict], x: float) -> int:
    for w in windows:
        if w['x_start'] <= x < w['x_end']:
            return int(w['index'])
    raise ValueError('x coordinate outside ROI windows')


def local_to_global(x: float, y: float, window: dict,
                    display_width: int, original_height: int) -> tuple[float, float]:
    """Map a click in scaled on-screen window to original full-ROI coordinates."""
    if display_width <= 0 or original_height <= 0:
        raise ValueError('Invalid display geometry')
    source_width = window['width']
    display_height = original_height * display_width / source_width
    if not (math.isfinite(x) and math.isfinite(y) and
            0 <= x < display_width and 0 <= y < display_height):
        raise ValueError('Click outside displayed window')
    return (window['x_start'] + x * source_width / display_width,
            y * original_height / display_height)


def global_to_local_markers(items: list[dict], window: dict) -> list[dict]:
    """Project event/reference/hint markers without changing original data."""
    out = []
    for item in items:
        x = float(item.get('x_pixel', -1))
        if window['x_start'] <= x < window['x_end']:
            local = dict(item)
            local['x_pixel'] = x - window['x_start']
            out.append(local)
    return out


def review_summary(windows: list[dict], reviewed: dict, references: list[dict]) -> dict[str, Any]:
    rows=[]
    for window in windows:
        idx = int(window['index'])
        marks = [p for p in references if window['x_start'] <= p['x_pixel'] < window['x_end']]
        rows.append({'window':idx+1, 'x_start':window['x_start'], 'x_end_exclusive':window['x_end'],
                     'reviewed':bool(reviewed.get(str(idx), False)), 'reference_count':len(marks)})
    complete = bool(rows and all(r['reviewed'] for r in rows))
    return {'windows': rows, 'reviewed_count':sum(r['reviewed'] for r in rows),
            'total_count':len(rows), 'all_windows_reviewed':complete,
            'reference_count':len(references)}
